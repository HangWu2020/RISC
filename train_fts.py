#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import model_ricn
import data_util
import utils
import loss

from tensorboardX import SummaryWriter

from hyper_ricn import *
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
parser.add_argument('-model_name', default='model_points_1.pth')
parser.add_argument('-restore', action='store_true', default=False)  # restore model_fts?
parser.add_argument('-fts_restore_root', default='./restore/fts_2023-10-30_03:57:07/')
parser.add_argument('-fts_model_name', default='model_fts_best_s6.pth')
parser.add_argument('-new_s2', action='store_true', default=True)
parser.add_argument('-create_fts', action='store_true', default=True)
parser.add_argument('-test', action='store_true', default=True)

args = parser.parse_args()

fts_path = os.path.join(args.restore_root, '%s_fts.hdf5'%args.model_name.split('.')[0])


'''---Save fts---'''
if not os.path.exists(fts_path) and args.new_s2 and args.create_fts:
    print ('Create fts_g')
    
    model_points = model_ricn.RICN(enc_features, dec_features, dec_features_fine, up_rate=1)
    model_points.cuda()
    checkpoint = torch.load(os.path.join(args.restore_root, args.model_name))
    model_points.load_state_dict(checkpoint['model_state_dict'])
    model_points.eval()
    start = checkpoint['epoch']
    print ('Loaded model %d'%start)
    
    dataset = data_util.dataset(args.dataset_root, 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size//2, shuffle=False)
    
    dataset_test = data_util.dataset(args.dataset_root, 'test')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size//2, shuffle=False)
    
    ft_g_all, ft_g_test_all = [], []
    
    loss_all_train = []
    
    for i, data in enumerate(dataloader, 0):
        points_xyz, gt_xyz = data[0].float().cuda(), data[1].float().cuda()
        
        with torch.no_grad():
            
            xyz_coarse, _, ft_g = model_points(points_xyz, scale, sigma, A, grid_pt, grid_r, inteval_comp_points, output_fts=True)
            
            ids = pn2_utils.furthest_point_sample(gt_xyz, 1024)
            ids = ids.unsqueeze(2).repeat(1, 1, 3).long()
            gt_xyz_coarse = torch.gather(gt_xyz, dim=1, index=ids)
            
            ft_g_all.append(ft_g.detach().cpu().numpy())
            
            loss_emd = loss.calc_emd(xyz_coarse, gt_xyz_coarse, eps=0.003, iterations=1000).mean()
            loss_all_train.append(loss_emd.item())
            
            print ('Train %d.%d'%(i, len(dataloader)))
    ft_g_all = np.concatenate(ft_g_all, axis=0)
    
    
    loss_all_test = []
    
    for i, data in enumerate(dataloader_test, 0):
        points_xyz, gt_xyz = data[0].float().cuda(), data[1].float().cuda()
        
        with torch.no_grad():
            
            xyz_coarse, _, ft_g = model_points(points_xyz, scale, sigma, A, grid_pt, grid_r, inteval_comp_points, output_fts=True)
            
            ids = pn2_utils.furthest_point_sample(gt_xyz, 1024)
            ids = ids.unsqueeze(2).repeat(1, 1, 3).long()
            gt_xyz_coarse = torch.gather(gt_xyz, dim=1, index=ids)
            
            ft_g_test_all.append(ft_g.detach().cpu().numpy())
            
            loss_emd = loss.calc_emd(xyz_coarse, gt_xyz_coarse, eps=0.003, iterations=1000).mean()
            loss_all_test.append(loss_emd.item())
            
            print ('Test %d.%d'%(i, len(dataloader_test)))
    ft_g_test_all = np.concatenate(ft_g_test_all, axis=0)
    
    print ('loss train: %.5f'%(np.mean(loss_all_train)*100))
    print ('loss test: %.5f'%(np.mean(loss_all_test)*100))
    
    fp = h5py.File(fts_path, 'w')
    fp.create_dataset('train_g', data=ft_g_all)
    fp.create_dataset('test_g', data=ft_g_test_all)
    fp.close()
    
    print ('fts saved, rerun code')
    sys.exit(0)
    
    
if args.new_s2 and not os.path.exists(fts_path):
    raise FileNotFoundError('%s not found'%fts_path)


'''1'''
if args.new_s2:
    dataset = data_util.dataset(args.dataset_root, 'train', fts_path=fts_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_fts, shuffle=True)
    dataset_test = data_util.dataset(args.dataset_root, 'test', fts_path=fts_path)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_fts, shuffle=False)
    
else:
    dataset = data_util.dataset(args.dataset_root, 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_test = data_util.dataset(args.dataset_root, 'test')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model_points = model_ricn.RICN(enc_features, dec_features, dec_features_fine, up_rate=1)
    model_points.cuda()
    checkpoint = torch.load(os.path.join(args.restore_root, args.model_name))
    model_points.load_state_dict(checkpoint['model_state_dict'])
    model_points.eval()
    start = checkpoint['epoch']
    print ('Loaded model %d'%start)

    
start = 0
res_pt = 64
res_r = 32
inteval_comp = 0.9

if args.new_s2:
    inv_features['bandwidths'] = [res_pt//2, res_pt//2, res_pt//2]
    inv_features['fmlp'][-1] = res_pt
else:
    inv_features['bandwidths'] = [grid_pt//2, grid_pt//2, grid_pt//2]
    inv_features['fmlp'][-1] = grid_pt

model_fts = model_ricn.Inverse(inv_features, args.new_s2)
model_fts.cuda()
    
optimizer_fts = torch.optim.Adam(model_fts.parameters(), lr=lr_fts)
# optimizer_fts = torch.optim.Adam(model_fts.parameters(), lr=lr_fts, weight_decay=1e-4)

if args.restore or args.test:
    save_folder = args.restore_root
    checkpoint = torch.load(os.path.join(args.fts_restore_root, args.fts_model_name))
    model_fts.load_state_dict(checkpoint['model_state_dict'])
    start = checkpoint['epoch']
    print ('Loaded model_fts %d'%start)
else:
    save_folder = os.path.join(args.save_root, datetime.now().strftime('fts_%Y-%m-%d_%H:%M:%S'))
    os.makedirs(os.path.join(save_folder))
    shutil.copy2('utils.py', os.path.join(save_folder, 'utils.py'))
    shutil.copy2('layers.py', os.path.join(save_folder, 'layers.py'))
    shutil.copy2('hyper_ricn.py', os.path.join(save_folder, 'hyper_ricn.py'))
    shutil.copy2('model_ricn.py', os.path.join(save_folder, 'model_ricn.py'))
    shutil.copy2('train_ricn.py', os.path.join(save_folder, 'train_ricn.py'))
    shutil.copy2('train_fts.py', os.path.join(save_folder, 'train_fts.py'))


if not args.test:
    writer = SummaryWriter(save_folder)
    
    best_loss = 10
    best_s4 = 0.0
    best_s6 = 0.0
    best_s16 = 0.0
    
    for epoch in range(start+1, num_epoch_fts+1):
        
        if (epoch > 0 and epoch % decay_step_fts == 0) or (epoch == start+1):
            lr = np.max([lr_fts*(decay_rate_fts**(epoch//decay_step_fts)), 1e-5])
            print ('Epoch %d, learning rate %.6f'%(epoch, lr))
            for param_group in optimizer_fts.param_groups:
                param_group['lr'] = lr
    
        loss_fts_all = []
        s_all = []
        
        model_fts.train()
        for i, data in enumerate(dataloader, 0):
            
            start_time = time.time()
            
            if args.new_s2:
                points_xyz, gt_xyz, fts_g = data
                points_xyz = points_xyz.float().cuda()
                ft_g = fts_g.float().cuda()
                
                part_l2, full_l2, loss_fts = model_fts(None, ft_g, points_xyz, scale, sigma, A, res_pt, res_r, inteval_comp)
                
            else:
                points_xyz, gt_xyz = data
                points_xyz = points_xyz.float().cuda()
                gt_xyz = gt_xyz.float().cuda()
                
                with torch.no_grad():
                    
                    
                    _, ft_grid, ft_g = model_points(points_xyz, scale, sigma, A, grid_pt, grid_r, inteval_comp_points, output_fts=True)
                    ft_grid = ft_grid.detach().contiguous()
                    ft_g = ft_g.detach().contiguous()
            
                part_l2, full_l2, loss_fts = model_fts(ft_grid, ft_g)
            
            optimizer_fts.zero_grad()
            loss_fts.backward()
            optimizer_fts.step()
            
            dist_all, batch_index, s = utils.feature_align(part_l2, full_l2, k=6)
            
            end_time = time.time()
        
            print ('Train epoch %d, %d.%d, eta %.2f min: %.5f'%(epoch, i, len(dataloader), (len(dataloader)-i)*(end_time-start_time)/60, loss_fts.item()), s)
            
            loss_fts_all.append(loss_fts.item())
            s_all.append(s)
            
        loss_fts_all = np.array(loss_fts_all)
        print ('Loss for epoch %d: %.4f'%(epoch, np.mean(loss_fts_all)))
        s_all = np.vstack(s_all)
        s1 = np.sum(s_all[:,0])
        s4 = np.sum(s_all[:,3])
        s6 = np.sum(s_all[:,5])
        print ('Average for epoch %d: %.4f, %.4f, %.4f\n'%(epoch, s1/len(dataset.gt_data), s4/len(dataset.gt_data), s6/len(dataset.gt_data)))
        
        writer.add_scalar('train_fts_loss', np.mean(loss_fts_all), global_step=epoch)
        writer.add_scalar('train_s4', s4/len(dataset.gt_data), global_step=epoch)
        writer.add_scalar('train_s6', s6/len(dataset.gt_data), global_step=epoch)
        
        
        '''------'''
        
        loss_fts_all = []
        s_all = []
        
        model_fts.eval()
        for i, data in enumerate(dataloader_test, 0):
                
            if args.new_s2:
                points_xyz, gt_xyz, fts_g = data
                points_xyz = points_xyz.float().cuda()
                ft_g = fts_g.float().cuda()
                
                with torch.no_grad():
                    part_l2, full_l2, loss_fts = model_fts(None, ft_g, points_xyz, scale, sigma, A, res_pt, res_r, inteval_comp)
                    dist_all, batch_index, s = utils.feature_align(part_l2, full_l2, res_pt, k=16)
                
            else:
                points_xyz, gt_xyz = data
                points_xyz = points_xyz.float().cuda()
                gt_xyz = gt_xyz.float().cuda()
                
                with torch.no_grad():
                    _, ft_grid, ft_g = model_points(points_xyz, scale, sigma, A, grid_pt, grid_r, inteval_comp_points, output_fts=True)
                    ft_grid = ft_grid.detach().contiguous()
                    ft_g = ft_g.detach().contiguous()
            
                    part_l2, full_l2, loss_fts = model_fts(ft_grid, ft_g)
            
                    dist_all, batch_index, s = utils.feature_align(part_l2, full_l2, grid_pt, k=16)
        
            print ('Test epoch %d, %d.%d'%(epoch, i, len(dataloader_test)))
            
            loss_fts_all.append(loss_fts.item())
            s_all.append(s)
            
        loss_fts_all = np.array(loss_fts_all)
        s_all = np.vstack(s_all)
        s1 = np.sum(s_all[:,0])
        s4 = np.sum(s_all[:,3])
        s6 = np.sum(s_all[:,5])
        s16 = np.sum(s_all[:,15])
        
        writer.add_scalar('test_fts_loss', np.mean(loss_fts_all), global_step=epoch)
        writer.add_scalar('test_s4', s4/len(dataset_test.gt_data), global_step=epoch)
        writer.add_scalar('test_s6', s6/len(dataset_test.gt_data), global_step=epoch)
        writer.add_scalar('test_s16', s16/len(dataset_test.gt_data), global_step=epoch)
        
        if np.mean(loss_fts_all) < best_loss:
            best_loss = np.mean(loss_fts_all)
            state = {'epoch': epoch, 'model_state_dict': model_fts.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_fts_best_loss.pth'))
            
        if s4 > best_s4:
            best_s4 = s4
            state = {'epoch': epoch, 'model_state_dict': model_fts.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_fts_best_s4.pth'))
            
        if s6 > best_s6:
            best_s6 = s6
            state = {'epoch': epoch, 'model_state_dict': model_fts.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_fts_best_s6.pth'))
            
        if s16 > best_s16:
            best_s16 = s16
            state = {'epoch': epoch, 'model_state_dict': model_fts.state_dict()}
            torch.save(state, os.path.join(save_folder, 'model_fts_best_s16.pth'))
            
            
else:
    batch_index_all = []
    loss_fts_all = []
    s_all = []
    
    model_fts.eval()
    for i, data in enumerate(dataloader_test, 0):
            
        if args.new_s2:
            points_xyz, gt_xyz, fts_g = data
            points_xyz = points_xyz.float().cuda()
            ft_g = fts_g.float().cuda()
            
            with torch.no_grad():
                part_l2, full_l2, loss_fts = model_fts(None, ft_g, points_xyz, scale, sigma, A, res_pt, res_r, inteval_comp)
                dist_all, batch_index, s = utils.feature_align(part_l2, full_l2, res_pt, k=16)
                
            
        else:
            points_xyz, gt_xyz = data
            points_xyz = points_xyz.float().cuda()
            gt_xyz = gt_xyz.float().cuda()
            
            with torch.no_grad():
                _, ft_grid, ft_g = model_points(points_xyz, scale, sigma, A, grid_pt, grid_r, inteval_comp_points, output_fts=True)
                ft_grid = ft_grid.detach().contiguous()
                ft_g = ft_g.detach().contiguous()
        
                part_l2, full_l2, loss_fts = model_fts(ft_grid, ft_g)
                
                dist_all, batch_index, s = utils.feature_align(part_l2, full_l2, grid_pt, k=16)
        
        print ('Test epoch %d, %d.%d'%(start, i, len(dataloader_test)))
        
        loss_fts_all.append(loss_fts.item())
        s_all.append(s)
        batch_index_all.append(batch_index)
        
    s_all = np.vstack(s_all)
    s1 = np.sum(s_all[:,0])
    s4 = np.sum(s_all[:,3])
    s6 = np.sum(s_all[:,5])
    s16 = np.sum(s_all[:,15])
    
    print('test_s4', s4/len(dataset_test.gt_data))
    print('test_s6', s6/len(dataset_test.gt_data))
    print('test_s16', s16/len(dataset_test.gt_data))
        
    batch_index_all_2 = torch.cat(batch_index_all, dim=0).detach().cpu().numpy()
    print (np.shape(batch_index_all_2))
    
    
