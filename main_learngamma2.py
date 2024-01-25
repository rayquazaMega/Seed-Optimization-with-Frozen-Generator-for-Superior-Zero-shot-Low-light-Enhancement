import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from pathlib2 import Path

try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    from skimage.measure import compare_psnr

from net import get_net
from net.VQVAE import VQVAE
from net.curve_model import SCurve
from net.decoder_model import DeepDecoder
from net.losses import HistEntropyLoss, FiedelityLoss, TVLoss, L_spa, HEP_Smooth_loss, ColorMapLoss, TVLoss_jit, BrightLoss
from net.noise import get_noise
from net.skip_model import SkipAdaDrop
#from net.scheduler import CustomScheduler
from utils.image_io import np_to_torch, torch_to_np, save_image, prepare_image, np_to_pil
from utils.imresize import np_imresize
from utils.gaussian import gaussian_blur

import time
#import pandas as pd
import clip

torch.manual_seed(384)
torch.cuda.manual_seed(384)
torch.cuda.manual_seed_all(384)

def downsample(image):
    return F.avg_pool2d(image, kernel_size=32, stride=16, padding=0)

class EngineModule(nn.Module):
    def __init__(self, input_path, output_dir, device, num_iter=15000, show_every=1000, drop_tau=0.1,
                 drop_mod_every=10000, num_inf_iter=100, input_depth=8, n_scale=5, lr=2e-3, illum_threshold=False):
        super(EngineModule, self).__init__()
        print(f"Processing {input_path}")

        self.output_dir = output_dir
        self.num_iter = num_iter
        self.show_every = show_every
        self.drop_mod_every = drop_mod_every
        self.num_inf_iter = num_inf_iter

        self.total_loss = None
        self.learning_rate = lr # 0.1
        self.illum_threshold = illum_threshold
        # init images
        self.image_name = Path(input_path).stem
        self.image = prepare_image(input_path)
        self.original_image = self.image.copy()
        factor = 1
        if self.image.shape[1] >= 800 or self.image.shape[2] >= 800:#while self.image.shape[1] >= 800 or self.image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            self.image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.image_torth = np_to_torch(self.image).float().to(device)
        self.illum_ref = downsample(self.image_torth.max(dim=1, keepdim=True)[0]).detach()
        self.fixed_illum = gaussian_blur(self.image_torth.max(dim=1, keepdim=True)[0], kernel_size=25, sigma=2.0)
        # init nets
        ref_VQVAE = VQVAE(need_dropout=False,need_sigmoid=True)
        ckpt = torch.load('/data3/gyx_tmp/vq-vae-2-pytorch-master/vqvae_560.pt')
        model_state_dict = ref_VQVAE.state_dict()
        for name, param in ckpt.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"Warning: Parameter '{name}' not found in the model.")
        self.reflect_net = ref_VQVAE.dec
        self.reflect_net = self.reflect_net.to(device)

        model_VQVAE = VQVAE(out_channels=1,need_sigmoid=True)
        '''model_state_dict = model_VQVAE.state_dict()
        for name, param in ckpt.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"Warning: Parameter '{name}' not found in the model.")'''
        self.illum_net = model_VQVAE.dec
        self.illum_net = self.illum_net.to(device)

        self.BNNet = torch.nn.BatchNorm2d(128)
        self.BNNet = self.BNNet.to(device)

        self.BNNet_ill = torch.nn.BatchNorm2d(128)
        self.BNNet_ill = self.BNNet_ill.to(device)

        # init inputs
        image_mean = self.image.mean()
        image_std = self.image.std()
        self.reflect_net_inputs = torch.randn((1, 128, self.image.shape[1]//4, self.image.shape[2]//4)).to('cuda')
        self.reflect_net_inputs  = self.reflect_net_inputs * image_std + image_mean

        self.illum_net_inputs = torch.randn((1, 128, self.image.shape[1]//4, self.image.shape[2]//4)).to('cuda')
        self.illum_net_inputs  = self.illum_net_inputs * image_std + image_mean

        self.gamma = torch.randn(1).to(device)
        self.gamma.requires_grad = True

        self.reflect_net_inputs.requires_grad = True
        for name,param in self.reflect_net.named_parameters():
            if not hasattr(param, 'is_bn'):  # 判断是否为 BatchNorm 层的参数
                param.requires_grad = False
        self.illum_net_inputs.requires_grad = True
        for name,param in self.illum_net.named_parameters():
            if not hasattr(param, 'is_bn'):  # 判断是否为 BatchNorm 层的参数
                param.requires_grad = False
        self.reflect_net.eval()
        self.illum_net.eval()

        # 662.43秒
        # init parameters
        #total_params = sum([p.numel() for p in self.illum_net.parameters()]+[p.numel() for p in self.reflect_net.parameters()])
        #print(f"Total parameters: {total_params / 1e6:.2f} million")
        self.parameters = [self.reflect_net_inputs] + \
                          [p for p in self.BNNet.parameters()] + \
                          [p for p in self.BNNet_ill.parameters()] + \
                          [self.illum_net_inputs] + \
                          [self.gamma]

    def forward(self):
        illum_net_input = self.BNNet_ill(self.illum_net_inputs)
        #illum_out = torch.mean(self.illum_net(illum_net_input),dim=1).unsqueeze(1).clamp(min=0.)*0.95+0.05
        illum_out = self.illum_net(illum_net_input).clamp(min=0.)

        reflect_net_input = self.BNNet(self.reflect_net_inputs)
        reflect_out = self.reflect_net(reflect_net_input)

        #illum_en,out = self.scurve_net(illum_out.clamp(min=0., max=1.).detach(), self.image_torth,ret_out=True)
        illum_en = illum_out**torch.sigmoid(self.gamma)

        image_en = reflect_out * illum_en
        image_out = reflect_out * illum_out
        return illum_out,illum_en,reflect_out,image_out,image_en,self.gamma

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of "Discrepant Untrained Network Priors"')
    parser.add_argument('--no_cuda', action='store_true', help='Use cuda?')
    parser.add_argument('--input_path', default='images/input3.png',
                        help='Path to input')
    parser.add_argument('--output_dir', default='output',
                        help='Path to save dir')
    parser.add_argument('--num_iter', type=int, default=15000,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--E', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--tv', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--mse', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--bri', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--ill', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--adj', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--illum_threshold', type=bool, default=False,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--show_every', type=int, default=1000,
                        help='How often to show the results (default: 1000)')
    parser.add_argument('--drop_tau', type=float, default=0.1,
                        help='Denoising stength for dropout ensemble (default: 0.1)')

    args = parser.parse_args()
    if (not args.no_cuda) and (not torch.cuda.is_available()):
        print("ERROR: cuda is not available, try running on CPU with option --no_cuda")
        sys.exit(1)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    engine = EngineModule(input_path=args.input_path, output_dir=args.output_dir, device=device, n_scale=5, input_depth=30, 
                        num_iter=args.num_iter, show_every=args.show_every, drop_tau=args.drop_tau,lr=args.lr,
                        illum_threshold=args.illum_threshold )
    fixed_illum = engine.fixed_illum
    image_torth = engine.image_torth
    eng_image = engine.image
    recon_criterion = FiedelityLoss().to(device)
    #jit_recon_criterion = torch.jit.script(recon_criterion)
    tv_criterion = TVLoss_jit(eng_image.shape[1],eng_image.shape[2]).to(device)
    #jit_tv_criterion = torch.jit.script(tv_criterion)
    adjillum_loss = BrightLoss(args.E).to(device)
    #adjillum_loss = HistEntropyLoss().to(device)
    #jit_bright_loss = torch.jit.script(adjillum_loss)

    #optimizer = torch.optim.Adam(engine.parameters, lr=engine.learning_rate)
    optimizer = torch.optim.Adam([
        {'params': engine.parameters, 'lr': engine.learning_rate},
    ])
    start_time = time.time()
    psnr = 0.
    #data = pd.DataFrame(columns=['image_name', 'step', 'scurve_out'])
    #engine = torch.jit.script(engine)
    for step in range(1, args.num_iter + 1):
        optimizer.zero_grad()
        illum_out,illum_en,reflect_out,image_out,image_en,gamma = engine()
        #recon_loss = recon_criterion(y=image_torth, y_pred=image_out)
        recon_loss = F.mse_loss(image_torth, image_out)
        bright_loss = F.mse_loss(illum_out, fixed_illum)

        adjill_loss = adjillum_loss(image_en)
        Ill_Smo_loss = tv_criterion(illum_out,reflect_out)
        tv_loss = tv_criterion(reflect_out)
        total_loss = args.mse*recon_loss + args.bri*bright_loss + args.ill * Ill_Smo_loss + args.adj * adjill_loss + args.tv * tv_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()

        #data = data._append({'image_name': engine.image_name, 'step': step, 'scurve_out': float(gamma.cpu())}, ignore_index=True)

        if step % args.show_every == 0:
            image_out_np = np.clip(torch_to_np(image_out.detach()), 0, 1)
            psnr = compare_psnr(eng_image, image_out_np)
            image_en_np = np.clip(torch_to_np(image_en.detach()), 0, 1)
            image_en_np = np_imresize(image_en_np, output_shape=engine.original_image.shape[1:])
            reflect_out_np = np.clip(torch_to_np(reflect_out.detach()), 0, 1)
            reflect_out_np = np_imresize(reflect_out_np, output_shape=engine.original_image.shape[1:])
            illum_out_np = np.clip(torch_to_np(illum_out.detach()), 0, 1)
            illum_out_np = np_imresize(illum_out_np, output_shape=engine.original_image.shape[1:])
            illum_en_np = np.clip(torch_to_np(illum_en.detach()), 0, 1)
            illum_en_np = np_imresize(illum_en_np, output_shape=engine.original_image.shape[1:])
            save_image(f"{engine.image_name}_out_{step}", image_en_np, args.output_dir)
            #save_image(f"{engine.image_name}_lowlight_{step}", image_out_np, args.output_dir)
            #save_image(f"{engine.image_name}_reflect_{step}", reflect_out_np, args.output_dir)
            #save_image(f"{engine.image_name}_illum_{step}", illum_out_np, args.output_dir)
            #save_image(f"{engine.image_name}_illum_en_{step}", illum_en_np, args.output_dir)


        if step % 8 == 0:
            print('Iteration: %05d    Loss: %.6f    BriLoss: %.6f    AdjLoss: %.6f' % (step, total_loss.item(), bright_loss.item(),adjill_loss.item()), '\r', end='')

    #data.to_csv(f'/data3/gyx_tmp/discrepant-untrained-nn-priors-master/results_ADIP/scurve_dir/{args.output_dir}_{engine.image_name}.csv', index=False)
    end_time = time.time()
    print('time: {} seconds'.format(end_time - start_time))

class Engine(object):
    def __init__(self, input_path, output_dir, device, num_iter=15000, show_every=1000, drop_tau=0.1,
                 drop_mod_every=10000, num_inf_iter=100, input_depth=8, n_scale=5, gamma=0.5, lr=2e-3,illum_threshold=False ):
        print(f"Processing {input_path}")

        self.output_dir = output_dir
        self.num_iter = num_iter
        self.show_every = show_every
        self.drop_mod_every = drop_mod_every
        self.num_inf_iter = num_inf_iter

        self.total_loss = None
        self.learning_rate = lr # 0.1
        self.gamma = gamma
        self.illum_threshold = illum_threshold
        # init images
        self.image_name = Path(input_path).stem
        self.image = prepare_image(input_path)
        self.original_image = self.image.copy()
        factor = 1
        if self.image.shape[1] >= 800 or self.image.shape[2] >= 800:#while self.image.shape[1] >= 800 or self.image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            self.image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.image_torth = np_to_torch(self.image).float().to(device)
        self.illum_ref = downsample(self.image_torth.max(dim=1, keepdim=True)[0]).detach()
        self.illum_gamma = downsample(self.image_torth.max(dim=1, keepdim=True)[0]**0.1).detach()
        self.fixed_illum = gaussian_blur(self.image_torth.max(dim=1, keepdim=True)[0], kernel_size=25, sigma=2.0)
        # init nets
        model_VQVAE = VQVAE(out_channels=1,need_sigmoid=True)
        self.illum_net = model_VQVAE.dec
        self.illum_net = self.illum_net.to(device)

        ref_VQVAE = VQVAE(need_dropout=False)
        ckpt = torch.load('/data3/gyx_tmp/vq-vae-2-pytorch-master/vqvae_560.pt')
        model_state_dict = ref_VQVAE.state_dict()
        for name, param in ckpt.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"Warning: Parameter '{name}' not found in the model.")
        self.reflect_net = ref_VQVAE.dec
        self.BNNet = torch.nn.BatchNorm2d(128)
        self.BNNet = self.BNNet.to(device)

        self.BNNet_ill = torch.nn.BatchNorm2d(128)
        self.BNNet_ill = self.BNNet_ill.to(device)
        self.reflect_net = self.reflect_net.to(device)

        #self.scurve_net = SCurve(size=self.image.shape)
        #self.scurve_net = self.scurve_net.to(device)

        # init inputs
        image_mean = self.image.mean()
        image_std = self.image.std()
        #self.reflect_net_inputs = torch.randn((1, 128, self.image.shape[1]//4, self.image.shape[2]//4)).to('cuda')
        #self.reflect_net_inputs = self.reflect_net_inputs = self.reflect_net_inputs * image_std + image_mean
        tmp_t,tmp_b,_,_ = ref_VQVAE.encode()
        self.reflect_net_inputs = torch.cat([ref_VQVAE.upsample_t(tmp_t),tmp_b], 1)

        self.illum_net_inputs = torch.randn((1, 128, self.image.shape[1]//4, self.image.shape[2]//4)).to('cuda')
        self.illum_net_inputs = self.reflect_net_inputs = self.reflect_net_inputs * image_std + image_mean

        self.reflect_net_inputs.requires_grad = True
        for name,param in self.reflect_net.named_parameters():
            if not hasattr(param, 'is_bn'):  # 判断是否为 BatchNorm 层的参数
                param.requires_grad = False
        self.illum_net_inputs.requires_grad = True
        for name,param in self.illum_net.named_parameters():
            if not hasattr(param, 'is_bn'):  # 判断是否为 BatchNorm 层的参数
                param.requires_grad = False


        # 662.43秒
        # init parameters
        self.parameters = [self.reflect_net_inputs] + \
                          [p for p in self.BNNet.parameters()] + \
                          [p for p in self.BNNet_ill.parameters()] + \
                          [p for p in self.scurve_net.parameters()] + \
                          [self.illum_net_inputs]

        # init loss
        self.recon_criterion = FiedelityLoss().to(device)
        self.jit_recon_criterion = torch.jit.script(self.recon_criterion)
        self.hist_criterion = HistEntropyLoss().to(device)
        self.tv_criterion = TVLoss_jit(self.image.shape[1],self.image.shape[2]).to(device)
        self.jit_tv_criterion = torch.jit.script(self.tv_criterion)
        self.bright_loss = BrightLoss(self.gamma).to(device)
        self.jit_bright_loss = torch.jit.script(self.bright_loss)
        self.spa_cirterion = L_spa().to(device)
        self.Hep_Smooth = HEP_Smooth_loss().to(device)
        self.colmap_cirterion = ColorMapLoss().to(device)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate*10)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.num_iter,eta_min=self.learning_rate)
        start_time = time.time()
        psnr = 0.
        for step in range(1, self.num_iter + 1):
            optimizer.zero_grad()
            
            #self.illum_net.activations = []

            illum_net_input = self.BNNet_ill(self.illum_net_inputs) #+ self.illum_net_inputs.clone().normal_() * self.illum_net.reg_std.data
            #print(illum_net_input.shape) # torch.Size([1, 16, 12, 18])
            illum_out = self.illum_net(illum_net_input).clamp(min=0.)

            reflect_net_input = self.reflect_net_inputs# + self.reflect_net_inputs.clone().normal_() * self.reflect_net.reg_std.data
            reflect_net_input = self.BNNet(reflect_net_input)
            reflect_out = self.reflect_net(reflect_net_input)

            illum_en = self.scurve_net(illum_out.clamp(min=0., max=1.).detach(), self.image_torth)

            image_en = reflect_out * illum_en#illum_en * reflect_out.detach()
            fixed_illum = self.fixed_illum
            #illum_out = fixed_illum
            image_out = reflect_out * illum_out#torch.log(reflect_out.clamp(min=4e-3)) + torch.log((illum_out+0.01))
            #print(torch.log(self.image_torth.clamp(min=1e-8)))

            recon_loss = self.jit_recon_criterion(y=self.image_torth, y_pred=image_out)
            if not self.illum_threshold:
                bright_loss = F.mse_loss(illum_out, fixed_illum)#F.mse_loss(downsample(illum_out), self.illum_ref)
            else:
                bright_loss = F.mse_loss(torch.where(illum_out > 0.9, torch.tensor(0.), illum_out), \
                torch.where(fixed_illum > 0.9, torch.tensor(0.), fixed_illum))

            # hist_loss = self.hist_criterion(image_en)
            adjill_loss = self.jit_bright_loss(illum_en=illum_en,illum_gamma=self.illum_gamma)
            # tv_loss = self.tv_criterion(reflect_out)
            # spa_loss = torch.mean(self.spa_cirterion(reflect_out,self.image_torth))
            Ill_Smo_loss = self.jit_tv_criterion(illum_out,reflect_out)
            #print(illum_out)
            # Hep_Smooth = self.Hep_Smooth(illum_out,reflect_out)
            # col_loss = self.colmap_cirterion(reflect_out,self.image_torth)
            #print(self.image_torth*255)
            self.total_loss = recon_loss + 0.05*bright_loss + 0.01 * Ill_Smo_loss + 0.01 * adjill_loss
            self.total_loss.backward(retain_graph=True)

            optimizer.step()
            #scheduler.step()

            # plot results and calculate PSNR
            if step % self.show_every == 0 and step >5000:
                image_out_np = np.clip(torch_to_np(image_out.detach()), 0, 1)
                psnr = compare_psnr(self.image, image_out_np)

                image_en_np = np.clip(torch_to_np(image_en.detach()), 0, 1)
                image_en_np = np_imresize(image_en_np, output_shape=self.original_image.shape[1:])

                reflect_out_np = np.clip(torch_to_np(reflect_out.detach()), 0, 1)
                reflect_out_np = np_imresize(reflect_out_np, output_shape=self.original_image.shape[1:])
                illum_out_np = np.clip(torch_to_np(illum_out.detach()), 0, 1)
                illum_out_np = np_imresize(illum_out_np, output_shape=self.original_image.shape[1:])
                illum_en_np = np.clip(torch_to_np(illum_en.detach()), 0, 1)
                illum_en_np = np_imresize(illum_en_np, output_shape=self.original_image.shape[1:])
                save_image(f"{self.image_name}_out_{step}", image_en_np, self.output_dir)
                save_image(f"{self.image_name}_lowlight_{step}", image_out_np, self.output_dir)
                save_image(f"{self.image_name}_reflect_{step}", reflect_out_np, self.output_dir)
                save_image(f"{self.image_name}_illum_{step}", illum_out_np, self.output_dir)
                save_image(f"{self.image_name}_illum_en_{step}", illum_en_np, self.output_dir)

            # obtain current result
            if step % 8 == 0:
                print('Iteration: %05d    Loss: %.6f    BriLoss: %.6f    PSNR: %.2f ' % (step, self.total_loss.item(), bright_loss.item(), psnr), '\r', end='')

        end_time = time.time()
        print('time: {} seconds'.format(end_time - start_time))

    def inference(self):
        with torch.no_grad():
            illum = self.illum_net(self.illum_net_inputs)
            illum_en = self.scurve_net(illum, self.image_torth)

            reflect_avg = None
            for step in range(self.num_inf_iter):
                reflect = self.reflect_net(self.reflect_net_inputs)
                if reflect_avg is None:
                    reflect_avg = reflect.detach()
                else:
                    reflect_avg = (reflect_avg * (step - 1) + reflect.detach()) / step

            image_en_avg = np.clip(torch_to_np((illum_en * reflect_avg).detach()), 0, 1)
            image_en_avg = np_imresize(image_en_avg, output_shape=self.original_image.shape[1:])
            save_image(f"{self.image_name}_out_final", image_en_avg, self.output_dir)

            print(f"Done. Please check the results in {self.output_dir}")


if __name__ == "__main__":
    main()
    '''parser = argparse.ArgumentParser(description='PyTorch implementation of "Discrepant Untrained Network Priors"')
    parser.add_argument('--no_cuda', action='store_true', help='Use cuda?')
    parser.add_argument('--input_path', default='images/input3.png',
                        help='Path to input')
    parser.add_argument('--output_dir', default='output',
                        help='Path to save dir')
    parser.add_argument('--num_iter', type=int, default=15000,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--illum_threshold', type=bool, default=False,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--show_every', type=int, default=1000,
                        help='How often to show the results (default: 1000)')
    parser.add_argument('--drop_tau', type=float, default=0.1,
                        help='Denoising stength for dropout ensemble (default: 0.1)')

    args = parser.parse_args()
    if (not args.no_cuda) and (not torch.cuda.is_available()):
        print("ERROR: cuda is not available, try running on CPU with option --no_cuda")
        sys.exit(1)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    engine = Engine(input_path=args.input_path, output_dir=args.output_dir, device=device, n_scale=5, input_depth=30, 
                        num_iter=args.num_iter, show_every=args.show_every, drop_tau=args.drop_tau,gamma=args.gamma,lr=args.lr,
                        illum_threshold=args.illum_threshold )
    engine.optimize()
    engine.inference()'''
