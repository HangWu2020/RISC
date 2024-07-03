#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import model_ricn
import data_util
import utils
import loss

from tensorboardX import SummaryWriter

import hyper_ricn as hyper
import numpy as np
import sys, os, time, shutil, h5py
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_root', default='./dataset')
parser.add_argument('-save_root', default='./restore')
parser.add_argument('-coarse_path', default='./restore/2023-10-23_21:42:31/')
parser.add_argument('-restore', action='store_true', default=False)
parser.add_argument('-restore_root', default='./restore/2023-10-17_01:58:20/')
parser.add_argument('-test', action='store_true', default=False)
args = parser.parse_args()

coarse_path = os.path.join(args.coarse_path, 'pretrained_coarse_points.hdf5')

dataset = data_util.dataset(args.dataset_root, 'train', coarse_path=coarse_path)
dataset_test = data_util.dataset(args.dataset_root, 'test', coarse_path=coarse_path)

'''1'''
model_refine = model_ricn.Decoder_fine()
model_refine.cuda()

if args.restore:
    checkpoint = torch.load(os.path.join(args.restore_root, 'model_refine_2.pth'))
    model_state_dict = checkpoint['model_state_dict']
    model_refine.load_state_dict(model_state_dict)
    start = checkpoint['epoch']
    print ('Loaded model_refine %d'%start)
    save_folder = args.restore_root
else:
    start = -1
    save_folder = os.path.join(args.save_root, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(os.path.join(save_folder))
    shutil.copy2('layers.py', os.path.join(save_folder, 'layers.py'))
    shutil.copy2('hyper_ricn.py', os.path.join(save_folder, 'hyper_ricn.py'))
    shutil.copy2('model_ricn.py', os.path.join(save_folder, 'model_ricn.py'))
    shutil.copy2('train_ricn.py', os.path.join(save_folder, 'train_ricn.py'))
    shutil.copy2('train_refine.py', os.path.join(save_folder, 'train_refine.py'))

if not args.test:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hyper.batch_size_refine, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=hyper.batch_size_refine, shuffle=False)
    
    optimizer_refine = torch.optim.Adam(model_refine.parameters(), lr=hyper.lr_refine)

    best_loss_emd = 10
    best_loss_cd_p = 10
    best_loss_cd_t = 10
    
    writer = SummaryWriter(save_folder)  # './restore/2023-10-03_01:27:37/'
    
    for epoch in range(start+1, hyper.num_epoch_refine+1):
        if (epoch > 0 and epoch % hyper.decay_step_refine == 0) or (epoch == start+1):
            lr = np.max([hyper.lr_refine*(hyper.decay_rate_refine**(epoch//hyper.decay_step_refine)), 1e-5])
            print ('Epoch %d, learning rate %.6f'%(epoch, lr))
            for param_group in optimizer_refine.param_groups:
                param_group['lr'] = lr
                
        '''train'''
        model_refine.train()
        
        loss_all_cd_1 = []
        loss_all_cd_2 = []
        
        for i, data in enumerate(dataloader, 0):
            
            start_time = time.time()
            
            inputs, gt, coarse = data
            inputs, gt, coarse = inputs.float().cuda(), gt.float().cuda(), coarse.float().cuda()
            # Optional data augmentation around (0, 90, 180, 270) for refinement
            inputs, gt = data_util.rotate_z_2(inputs, gt, init_angle=None, random_rotate=True)
            
            skeleton_xyz, coarse_xyz, refine_xyz = model_refine(inputs, coarse, k=4)
            
            cd_0, _ = loss.calc_cd(skeleton_xyz, gt)
            cd_1, _ = loss.calc_cd(coarse_xyz, gt)
            cd_2, _ = loss.calc_cd(refine_xyz, gt)
                
            loss_all = cd_0.mean() + cd_1.mean() + cd_2.mean()
            
            optimizer_refine.zero_grad()
            loss_all.backward()
            optimizer_refine.step()
            
            loss_all_cd_1.append(cd_1.mean().item())
            loss_all_cd_2.append(cd_2.mean().item())
            
            end_time = time.time()
            
            print('Epoch %02d.%d, eta %.2f min: loss: %.3f, %.3f'%(epoch, i, (len(dataloader)-i)*(end_time-start_time)/60, cd_1.mean().item()*1e4, cd_2.mean().item()*1e4))
            
        writer.add_scalar('train_cd_1', np.mean(loss_all_cd_1), global_step=epoch)
        writer.add_scalar('train_cd_2', np.mean(loss_all_cd_2), global_step=epoch)
        
                
        '''test'''
        model_refine.eval()
        
        loss_all_emd = []
        loss_all_cd_p = []
        loss_all_cd_t = []
        
        for i, data in enumerate(dataloader_test, 0):
            
            start_time = time.time()
            
            inputs, gt, coarse = data
            inputs, gt, coarse = inputs.float().cuda(), gt.float().cuda(), coarse.float().cuda()
            
            with torch.no_grad():
                skeleton_xyz, coarse_xyz, refine_xyz = model_refine(inputs, coarse, k=4)
                
                loss_emd = loss.calc_emd(refine_xyz, gt, eps=0.004, iterations=3000).mean()
                loss_cd_p, loss_cd_t = loss.calc_cd(refine_xyz, gt)
                
                loss_all_emd.append(loss_emd.detach().cpu().numpy())
                loss_all_cd_p.append(loss_cd_p.detach().cpu().numpy())
                loss_all_cd_t.append(loss_cd_t.detach().cpu().numpy())
            
            end_time = time.time()
            
            print('Test %02d.%d, eta %.2f min'%(epoch, i, (len(dataloader_test)-i)*(end_time-start_time)/60))
        
        loss_all_emd = np.hstack(loss_all_emd)
        loss_all_cd_p = np.hstack(loss_all_cd_p)
        loss_all_cd_t = np.hstack(loss_all_cd_t)
        
        writer.add_scalar('test_emd', np.mean(loss_all_emd), global_step=epoch)
        writer.add_scalar('test_cd_p', np.mean(loss_all_cd_p), global_step=epoch)
        writer.add_scalar('test_cd_t', np.mean(loss_all_cd_t), global_step=epoch)
        
        print ('test_emd', np.mean(loss_all_emd)*100, 'test_cd_p', np.mean(loss_all_cd_p)*100)
        
        if np.mean(loss_all_emd)<best_loss_emd:
            best_loss_emd = np.mean(loss_all_emd)
            state = {'epoch': epoch, 'model_state_dict': model_refine.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_refine_1.pth'))
        
        if np.mean(loss_all_cd_p)<best_loss_cd_p:
            best_loss_cd_p = np.mean(loss_all_cd_p)
            state = {'epoch': epoch, 'model_state_dict': model_refine.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_refine_2.pth'))

        if np.mean(loss_all_cd_t)<best_loss_cd_t:
            best_loss_cd_t = np.mean(loss_all_cd_t)
            state = {'epoch': epoch, 'model_state_dict': model_refine.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_refine_3.pth'))
        

else:
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=hyper.batch_size_refine, shuffle=False)
    
    model_refine.eval()
    
    loss_all_emd = []
    loss_all_cd_p = []
    loss_all_cd_t = []
    
    for i, data in enumerate(dataloader_test, 0):
        
        start_time = time.time()
        
        inputs, gt, coarse = data
        inputs, gt, coarse = inputs.float().cuda(), gt.float().cuda(), coarse.float().cuda()
        
        with torch.no_grad():
            coarse_xyz, refine_xyz = model_refine(inputs, coarse, k=8)
        
            loss_emd = loss.calc_emd(refine_xyz, gt, eps=0.004, iterations=3000).mean()
            loss_cd_p, loss_cd_t = loss.calc_cd(refine_xyz, gt)
            
            loss_all_emd.append(loss_emd.detach().cpu().numpy())
            loss_all_cd_p.append(loss_cd_p.detach().cpu().numpy())
            loss_all_cd_t.append(loss_cd_t.detach().cpu().numpy())
        
        end_time = time.time()
        
        print('Test %d, eta %.2f min'%(i, (len(dataloader_test)-i)*(end_time-start_time)/60))
    
    loss_all_emd = np.hstack(loss_all_emd)
    loss_all_cd_p = np.hstack(loss_all_cd_p)
    loss_all_cd_t = np.hstack(loss_all_cd_t)
    
    print ('emd: %.4f; cd_p: %.4f; cd_t: %.4f'%(np.mean(loss_all_emd)*100, np.mean(loss_all_cd_p), np.mean(loss_all_cd_t)))
