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

sys.path.append('./extension/pointnet2/')
import pointnet2_utils as pn2_utils

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_root', default='./dataset')
parser.add_argument('-save_root', default='./restore')
parser.add_argument('-restore_root', default='./restore/2023-10-26_23:56:28/')
parser.add_argument('-restore', action='store_true', default=True)
parser.add_argument('-test', action='store_true', default=True)
args = parser.parse_args()


'''1'''
os.mkdir(args.save_root) if not os.path.exists(args.save_root) else None
model_points = model_ricn.RICN(hyper.enc_features, hyper.dec_features, hyper.dec_features_fine, up_rate=1)
model_points.cuda()

optimizer_points = torch.optim.Adam(model_points.parameters(), lr=hyper.lr_points)

if args.restore:
    checkpoint = torch.load(os.path.join(args.restore_root, 'model_points_1.pth'))
    model_state_dict = checkpoint['model_state_dict']
    ### deprecated code ###
    # del_key = []
    # for key, _ in list(model_state_dict.items()):
    #     if 'decoder_fine' in key:
    #         del_key.append(key)
    # for key in del_key:
    #     del model_state_dict[key]
    ### --------------- ###
    model_points.load_state_dict(model_state_dict)
    start = checkpoint['epoch']
    print ('Loaded model_points %d'%start)
    save_folder = args.restore_root  
else:
    start = 0
    save_folder = os.path.join(args.save_root, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(os.path.join(save_folder))
    shutil.copy2('layers.py', os.path.join(save_folder, 'layers.py'))
    shutil.copy2('hyper_ricn.py', os.path.join(save_folder, 'hyper_ricn.py'))
    shutil.copy2('model_ricn.py', os.path.join(save_folder, 'model_ricn.py'))
    shutil.copy2('train_ricn.py', os.path.join(save_folder, 'train_ricn.py'))

dataset = data_util.dataset(args.dataset_root, 'train')
dataset_test = data_util.dataset(args.dataset_root, 'test')

'''2'''
if not args.test:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hyper.batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=hyper.batch_size, shuffle=False)
    
    best_loss = 10
    writer = SummaryWriter(save_folder)  # './restore/2023-10-03_01:27:37/'
    
    for epoch in range(start+1, hyper.num_epoch+1):
        if (epoch > 0 and epoch % hyper.decay_step == 0) or (epoch == start+1):
            lr = np.max([hyper.lr_points*(hyper.decay_rate**(epoch//hyper.decay_step)), 1e-5])
            print ('Epoch %d, learning rate %.6f'%(epoch, lr))
            for param_group in optimizer_points.param_groups:
                param_group['lr'] = lr
        
        '''train'''
        model_points.train()
        
        loss_all = []
        
        for i, data in enumerate(dataloader, 0):
            
            start_time = time.time()
            
            points_xyz, gt_xyz = data
            points_xyz, gt_xyz = points_xyz.float().cuda(), gt_xyz.float().cuda()
            
            ids = pn2_utils.furthest_point_sample(gt_xyz, 1024)  # (b, 1024)
            ids = ids.unsqueeze(2).repeat(1, 1, 3).long()  # (b, 1024, 3)
            gt_xyz_coarse = torch.gather(gt_xyz, dim=1, index=ids)  # (b, 1024, 3)
            
            xyz_coarse = model_points(points_xyz, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points)
            
            # data_util.view_pcd(xyz_fine[7].detach().cpu().numpy())
            
            loss_coarse = loss.calc_emd(xyz_coarse, gt_xyz_coarse, eps=0.005, iterations=100).mean()
            loss_all.append(loss_coarse.item())
            
            optimizer_points.zero_grad()
            loss_coarse.backward()
            optimizer_points.step()
            
            end_time = time.time()
            
            print('Epoch %02d.%d, eta %.2f min: loss: %.3f'%(epoch, i, (len(dataloader)-i)*(end_time-start_time)/60, loss_coarse.item()*1e2))
            
        writer.add_scalar('train_coarse', np.mean(loss_all), global_step=epoch)
        
        
        '''test'''
        model_points.eval()
        
        loss_all = []
        
        for i, data in enumerate(dataloader_test, 0):
            
            start_time = time.time()
                    
            points_xyz, gt_xyz = data
            points_xyz, gt_xyz = points_xyz.float().cuda(), gt_xyz.float().cuda()
            
            with torch.no_grad():
                xyz_coarse = model_points(points_xyz, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points)
                
                # points_xyz_rot, _ = data_util.rotate_z(points_xyz, None, init_angle=None, random_rotate=True)
                # xyz_coarse = model_points(points_xyz_rot, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points)
                # data_util.view_pcd(xyz_fine[6].detach().cpu().numpy())
                
                ids = pn2_utils.furthest_point_sample(gt_xyz, 1024)  # (b, 512)
                ids = ids.unsqueeze(2).repeat(1, 1, 3).long()  # (b, 512, 3)
                gt_xyz_coarse = torch.gather(gt_xyz, dim=1, index=ids)  # (b, 512, 3)
                
                loss_coarse = loss.calc_emd(xyz_coarse, gt_xyz_coarse, eps=0.003, iterations=1000)
                loss_all.append(loss_coarse.detach().cpu().numpy())
                
            end_time = time.time()
                
            print('Test epoch %02d.%d, eta %.2f min:'%(epoch, i, (len(dataloader_test)-i)*(end_time-start_time)/60))
        
        loss_all = np.hstack(loss_all)
        
        writer.add_scalar('test_emd_coarse', np.mean(loss_all), global_step=epoch)
        
        if np.mean(loss_all)<best_loss:
            best_loss = np.mean(loss_all)
            state = {'epoch': epoch, 'model_state_dict': model_points.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_points_1.pth'))
    
    writer.close()
    
    
            
else:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=hyper.batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=hyper.batch_size, shuffle=False)
    
    model_points.eval()
    
    loss_all, points_all_test = [], []
    for i, data in enumerate(dataloader_test, 0):
        
        if i >= 0:
        
            start_time = time.time()
                    
            points_xyz, gt_xyz = data
            points_xyz, gt_xyz = points_xyz.float().cuda(), gt_xyz.float().cuda()
            
            with torch.no_grad():
                xyz_coarse = model_points(points_xyz, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points)
                
                # points_xyz_rot, _ = data_util.rotate_z(points_xyz, None, init_angle=None, random_rotate=True)
                # xyz_coarse = model_points(points_xyz_rot, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points)
                # data_util.view_pcd(points_xyz_rot[0].detach().cpu().numpy())
                # data_util.view_pcd(xyz_coarse[5].detach().cpu().numpy())
                
                ids = pn2_utils.furthest_point_sample(gt_xyz, 1024)  # (b, 512)
                ids = ids.unsqueeze(2).repeat(1, 1, 3).long()  # (b, 512, 3)
                gt_xyz_coarse = torch.gather(gt_xyz, dim=1, index=ids)  # (b, 512, 3)
                
                loss_coarse = loss.calc_emd(xyz_coarse, gt_xyz_coarse, eps=0.003, iterations=1000)
                loss_all.append(loss_coarse.detach().cpu().numpy())
                
                points_all_test.append(xyz_coarse.detach().cpu().numpy())
                
            end_time = time.time()
                
            print('Test epoch %02d.%d, eta %.2f min'%(start, i, (len(dataloader_test)-i)*(end_time-start_time)/60))
    
    loss_all_test = np.hstack(loss_all)
    points_all_test = np.concatenate(points_all_test)
    print ('test loss: %.4f'%(np.mean(loss_all_test)*100))
    
    
    ## np.save('points_all_test_2.npy', points_all_test)
    ## points_all_test = np.load('points_all_test.npy')
    
    # points_path = os.path.join(args.restore_root, 'pretrained_coarse_points.hdf5')
    # fp = h5py.File(points_path, 'w')
    # fp.create_dataset('pretrained_train', data=points_all_train)
    # fp.create_dataset('pretrained_test', data=points_all_test)
    # fp.close()
    
    
    # data_util.view_pcd(points_all_train[200])
    # data_util.view_pcd(points_all_test[7000])
    
