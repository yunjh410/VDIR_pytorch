import torch
import os, glob
import numpy as np
import argparse
from utils import *
from model import *
from dataset import AWGN_Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from itertools import chain


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./DIV2K', help='data dir')
    parser.add_argument('--gpu',type=int ,default='0', help='gpu num')
    parser.add_argument('--mode',type=str ,default='AWGN', help='test mode')
    parser.add_argument('--max_epoch',type=int ,default='10', help='max train epoch')
    parser.add_argument('--batch_size',type=int ,default='1', help='training batch size')
    parser.add_argument('--num_workers',type=int ,default='4', help='number of workers')
    parser.add_argument('--exp_name', type=str, default='test', help='the name of experiment, where path file saved')
    parser.add_argument('--sigma', type=int, default=10)
    
    opt = parser.parse_args()
    
        
    # make path to model save
    # os.makedirs(f'./saved_models/{opt.exp_name}')
    
    # GPU settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:" + str(opt.gpu))
    torch.cuda.set_device(device)

    ### Dataset
    test_dataset = AWGN_Dataset(opt, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))
    
    ## Model
    model = VDIR(channels=64)
    model = model.cuda()
    
    # get checkpoint adrs
    saved_model_adrs = sorted(glob.glob(f'./saved_models/{opt.exp_name}/*.pth'))
    if len(saved_model_adrs):
        # saved_model = f'./saved_models/{opt.exp_name}/015.pth'
        saved_model = saved_model_adrs[-1]
    else:
        raise Exception(f'no saved weights in ./saved_models/{opt.exp_name}/')
    
    checkpoint = torch.load(saved_model, map_location=torch.device(f'cuda:{opt.gpu}'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test(opt, test_loader, model)


def test(opt, test_loader, model):
    
    model.eval()
    psnr_ = list()
    with torch.no_grad():
        for batch_idx, (images, labels, sigmas_noise) in enumerate(test_loader):
            images, labels, sigmas_noise = images.cuda(), labels.cuda(), sigmas_noise.cuda()
            
            # forward prop
            mu, log_sigma_sq, y_hat, x, sigma_noise_est= model(images)
            
            labels = labels.detach().cpu().numpy() * 255
            x = x.detach().cpu().numpy() * 255
            x = np.clip(x, 0, 255)
            psnr_.append(psnr(x, labels))
    
    print(np.mean(psnr_))

if __name__ == '__main__':
    main()
