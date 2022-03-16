import torch
import os, time
import numpy as np
import argparse
import torchsummary
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
    parser.add_argument('--mode',type=str ,default='AWGN', help='training mode')
    parser.add_argument('--max_epoch',type=int ,default='10', help='max train epoch')
    parser.add_argument('--batch_size',type=int ,default='32', help='training batch size')
    parser.add_argument('--num_workers',type=int ,default='4', help='number of workers')
    parser.add_argument('--exp_name', type=str, default='test3', help='the name of experiment, where path file saved')
    
    opt = parser.parse_args()
    
    opt.gpu = 2
    opt.exp_name ='jpeg_noise_removal'
    # make path to model save
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    
    # GPU settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:" + str(opt.gpu))
    torch.cuda.set_device(device)
    
    log = logger(f'./saved_models/{opt.exp_name}/train_log.txt', 'train', 'a')
    opt_log = '------------ Options -------------\n'
    for k, v in vars(opt).items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    log.info(opt_log)

    ### Dataset
    train_dataset = AWGN_Dataset(opt, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))
    log.info(f'training : {len(train_dataset)}')
    
    ## Model
    model = VDIR(channels=64)
    Dis = Discriminator(channels=64)
    model = model.cuda()
    Dis = Dis.cuda()


    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(params=Dis.parameters(), lr=2e-4, betas=(0.5, 0.999))    
    start = time.time()
    # torchsummary.summary(model, input_size = (3,96,96))

    for epoch in range(opt.max_epoch):
    
        loss, loss_denoise, loss_KL, loss_g, loss_ae_recon, loss_EST, loss_d = train(opt=opt,
                                                                            model=model,
                                                                            Discriminator=Dis,
                                                                            train_loader=train_loader,
                                                                            optimizer=optimizer,
                                                                            optimizer_d=optimizer_d,
                                                                            epoch=epoch)
        elapsed_time = date_time(time.time() - start)
        log.info(f'Elapsed time: {elapsed_time}')
        # learning rate decay exponentially to half every 5 epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * (0.5 ** 0.2), 2e-5)
        for param_group in optimizer_d.param_groups:
            param_group['lr'] = max(param_group['lr'] * (0.5 ** 0.2), 2e-5)
        
        ## validate?
        save_checkpoint(f'./saved_models/{opt.exp_name}/{str(epoch).zfill(3)}.pth', model, Dis, optimizer, optimizer_d, epoch)
        log.info(f'[{epoch}/{opt.max_epoch}] Train Loss: {loss}\n')
        log.info(f'Denoise loss: {loss_denoise}, KL loss: {loss_KL}, Gen loss: {loss_g}, AE recon loss: {loss_ae_recon}, Noise level loss: {loss_EST}, Discrim loss: {loss_d}')


def train(opt, train_loader, model, Discriminator, optimizer, optimizer_d, epoch):
    
    model.train()
    Discriminator.train()
    
    losses = AverageMeter()
    losses_d = AverageMeter()
    losses_g = AverageMeter()
    losses_denoise = AverageMeter()
    losses_KL = AverageMeter()
    losses_EST = AverageMeter()
    losses_ae_recon = AverageMeter()
    
    l1_loss = nn.L1Loss()

    for batch_idx, (images, labels, sigmas_noise) in enumerate(train_loader):
        images, labels, sigmas_noise = images.cuda(), labels.cuda(), sigmas_noise.cuda()
        
        # forward prop
        mu, log_sigma_sq, y_hat, x, sigma_noise_est= model(images)
        
        # update Discriminator first
        f_logits_d = Discriminator(y_hat)
        r_logits_d = Discriminator(images)
        
        # adversarial loss for discriminator
        loss_d_fake = F.binary_cross_entropy_with_logits(input=f_logits_d, target=torch.zeros_like(f_logits_d)).mean()
        loss_d_real = F.binary_cross_entropy_with_logits(input=r_logits_d, target=torch.ones_like(r_logits_d)).mean()
        loss_d = loss_d_fake + loss_d_real
        losses_d.update(loss_d.item(), images.size(0))
        
        optimizer_d.zero_grad()
        loss_d.backward(retain_graph = True)
        optimizer_d.step()
        
        # update model
        f_logits_g = Discriminator(y_hat)
                
        # calculate losses
        # First term
        loss_denoise = l1_loss(x, labels)
        losses_denoise.update(loss_denoise.item(), images.size(0))
        
        # Second term, KL Divergence Loss
        loss_KL = 0.5 * (torch.exp(log_sigma_sq) + torch.square(mu) - 1 - log_sigma_sq).mean()
        losses_KL.update(loss_KL.item(), images.size(0))
        
        # Third Term
        # l1 loss between decoder output and model input
        loss_AE_recon = l1_loss(y_hat, images)
        losses_ae_recon.update(loss_AE_recon.item(), images.size(0))
        
        # l1 loss between sigma and estimated sigma
        loss_EST = l1_loss(sigmas_noise, sigma_noise_est) 
        losses_EST.update(loss_EST.item(), images.size(0))
        
        # adversarial loss for generator
        loss_g = F.binary_cross_entropy_with_logits(input=f_logits_g, target=torch.ones_like(f_logits_g)).mean()
        losses_g.update(loss_g.item(), images.size(0))
        
        loss = loss_denoise + 0.01*loss_KL + loss_AE_recon + 0.001*loss_g + loss_EST
        losses.update(loss.item(), images.size(0))
        
        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 1000 == 0:
            print(f'Epoch: [{epoch}/{opt.max_epoch}] [{batch_idx}/{len(train_loader)}]')
            print(f'Train Loss : {losses.avg:0.3f} ({losses.val:0.3f}), Denoise loss: {losses_denoise.avg:0.3f} ({losses_denoise.val:0.3f}), KL loss: {losses_KL.avg:0.3f} ({losses_KL.val:0.3f}),')
            print(f'Gen loss: {losses_g.avg:0.3f} ({losses_g.val:0.3f}), \
                    AE recon loss: {losses_ae_recon.avg:0.3f} ({losses_ae_recon.val:0.3f}), \
                    Noise level loss: {losses_EST.avg:0.3f} ({losses_EST.val:0.3f}), \
                    Discrim loss: {losses_d.avg:0.3f} ({losses_d.val:0.3f})')
         
    return losses.avg, losses_denoise.avg, losses_KL.avg, losses_g.avg, losses_ae_recon.avg, losses_EST.avg, losses_d.avg

if __name__ == '__main__':
    main()
