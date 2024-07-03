#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import model_ricn, data_util, loss, utils, fts_align
import hyper_ricn as hyper
import numpy as np
import sys, os, time, shutil, h5py
from datetime import datetime
import argparse

sys.path.append('./extension/pointnet2/')
import pointnet2_utils as pn2_utils

parser = argparse.ArgumentParser()
parser.add_argument('-dataset_root', default='../RICN_2/dataset')
parser.add_argument('-ricn_root', default='./restore/2023-10-26_23:56:28/')
parser.add_argument('-fts_root', default='./restore/fts_2023-10-30_03:57:07/')
parser.add_argument('-refine_root', default='./restore/2023-11-02_00:44:58/')  # '2023-11-02_00:44:58''2023-11-06_13:58:37/'
parser.add_argument('-save_type', default='inv')
parser.add_argument('-batch_size', type=int, default=6)

args = parser.parse_args()



'''1. load models'''
## model_points ##
model_points = model_ricn.RICN(hyper.enc_features, hyper.dec_features, hyper.dec_features_fine, up_rate=1).cuda()
model_points.eval()

checkpoint = torch.load(os.path.join(args.ricn_root, 'model_points_1.pth'))
model_state_dict = checkpoint['model_state_dict']
model_points.load_state_dict(model_state_dict)
print ('Loaded model_points %d'%checkpoint['epoch'])


## model_fts ##
model_fts = model_ricn.Inverse(hyper.inv_features, True).cuda()
model_fts.eval()

checkpoint = torch.load(os.path.join(args.fts_root, 'model_fts_best_s6.pth'))
model_fts.load_state_dict(checkpoint['model_state_dict'])
print ('Loaded model_fts %d'%checkpoint['epoch'])


## model_refine ##
model_refine = model_ricn.Decoder_fine().cuda()
model_refine.eval()

checkpoint = torch.load(os.path.join(args.refine_root, 'model_refine_1.pth'))
model_state_dict = checkpoint['model_state_dict']
model_refine.load_state_dict(model_state_dict)
print ('Loaded model_refine %d'%checkpoint['epoch'])


'''2. load dataset'''
dataset_test = data_util.dataset(args.dataset_root, 'test')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

loss_all, points_all_test, batch_index_all, s_all = [], [], [], []
loss_all_emd, loss_all_cd_p, loss_all_cd_t = [], [], []
points_all = []


for i, data in enumerate(dataloader_test, 0):
    
    if i >= 0:
    
        start_time = time.time()
        
        points_xyz, gt_xyz = data[0].float().cuda(), data[1].float().cuda()
        
        angles = np.random.uniform(0, 2*np.pi, args.batch_size)
        points_xyz_rot, gt_xyz_rot = data_util.rotate_z(points_xyz, gt_xyz, init_angle=angles, random_rotate=False)
        
        with torch.no_grad():
            ### rotate ###
            xyz_coarse_rot, ft_grid_rot, ft_g_rot = model_points(points_xyz_rot, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points, output_fts=True)
            part_l2_rot, full_l2_rot, loss_fts_rot = model_fts(ft_grid_rot, ft_g_rot, points_xyz_rot, hyper.scale, hyper.sigma, hyper.A, hyper.grid_pt, hyper.grid_r, hyper.inteval_comp_points)
            dist_all, batch_index, s = utils.feature_align(full_l2_rot, part_l2_rot, hyper.grid_pt, k=16)
            
            poses = []
            for j in range(args.batch_size):
                pose = fts_align.feature_align(batch_index[j].cpu().numpy(), hyper.grid_pt, points_xyz_rot[j:j+1], xyz_coarse_rot[j:j+1], iter_step=6)
                poses.append(pose)
            poses = np.array(poses)
            # print (poses)
            
            points_xyz_bk_rotate, _ = data_util.rotate_z(points_xyz_rot, None, init_angle=-poses/64*2*np.pi, random_rotate=False)
            skeleton_xyz, coarse_xyz, refine_xyz = model_refine(points_xyz_bk_rotate, xyz_coarse_rot, k=4)
            
            idx = np.arange(1024)
            np.random.shuffle(idx)
            if args.save_type == 'equ':
                idx = idx[0:32]
            if args.save_type == 'inv':
                idx = idx[0:128]
            xyz_coarse_rot = xyz_coarse_rot[:, idx, :].contiguous()
            refine_xyz = torch.cat([refine_xyz, xyz_coarse_rot], dim=1)  # (b, 2048+n_merge, 3)
            ids = pn2_utils.furthest_point_sample(refine_xyz, 2048).unsqueeze(2).repeat(1, 1, 3).long()  # (b, m, 3)
            refine_xyz = torch.gather(refine_xyz, dim=1, index=ids).contiguous()  # (b, 2048, 3)
            
            refine_xyz_rot, _ = data_util.rotate_z(refine_xyz, None, init_angle=poses/64*2*np.pi, random_rotate=False)
            
            if args.save_type == 'equ':
                loss_emd = loss.calc_emd(refine_xyz_rot, gt_xyz_rot, eps=0.004, iterations=3000)
                loss_cd_p, loss_cd_t = loss.calc_cd(refine_xyz_rot, gt_xyz_rot)
                points_out, _ = data_util.rotate_z(refine_xyz_rot, None, init_angle=-angles, random_rotate=False)
                points_out = points_out.detach().cpu().numpy()
                points_all.append(points_out)
                
            if args.save_type == 'inv':
                loss_emd = loss.calc_emd(refine_xyz, gt_xyz, eps=0.004, iterations=3000)
                loss_cd_p, loss_cd_t = loss.calc_cd(refine_xyz, gt_xyz)
                points_all.append(refine_xyz.detach().cpu().numpy())
            
            loss_all_emd.append(loss_emd.detach().cpu().numpy())
            loss_all_cd_p.append(loss_cd_p.detach().cpu().numpy())
            loss_all_cd_t.append(loss_cd_t.detach().cpu().numpy())
            
            # data_util.view_pcd(refine_xyz[4].detach().cpu().numpy())
            # data_util.view_pcd(gt_xyz[4].detach().cpu().numpy())
            # data_util.view_pcd(points_all[-1][4])
            
            end_time = time.time()
            print('Test %d, eta %.2f min'%(i, (len(dataloader_test)-i)*(end_time-start_time)/60))
            
            
            
loss_all_emd = np.hstack(loss_all_emd)
loss_all_cd_p = np.hstack(loss_all_cd_p)
loss_all_cd_t = np.hstack(loss_all_cd_t)

print ('emd: %.4f; cd_p: %.4f; cd_t: %.4f'%((np.mean(loss_all_emd)*100), (np.mean(loss_all_cd_p)*100), (np.mean(loss_all_cd_t))*100))
        
points_all = np.concatenate(points_all, axis=0)
np.save('ours_%s_merge.npy'%args.save_type, points_all)
