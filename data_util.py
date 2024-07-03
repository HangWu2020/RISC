#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import open3d as o3d
from matplotlib import cm
import os, sys
import h5py


def rotate_z (inputs, gt, init_angle=0.0, random_rotate=True, descrete_rotate=None):
    batch_size = inputs.shape[0]
    if random_rotate is True:
        assert init_angle is None, 'init_angle should be set None when random_rotate is True'
        if descrete_rotate is None:
            angles = np.random.uniform(0, 2*np.pi, batch_size)
        else:
            angles = np.random.randint(descrete_rotate, size=batch_size)/descrete_rotate*2*np.pi  # descrete_rotate=64   
    else:
        angles = init_angle * np.ones(batch_size, dtype=np.float32)
        
    R = np.array([[np.cos(angles),       -np.sin(angles),        np.zeros_like(angles)],
                  [np.sin(angles),        np.cos(angles),        np.zeros_like(angles)],
                  [np.zeros_like(angles), np.zeros_like(angles), np.ones_like(angles)]], dtype=np.float32)
    R = torch.tensor(R, device=inputs.device).permute(2, 0, 1).contiguous()
    
    inputs = inputs.permute(0, 2, 1).contiguous()
    points_out = torch.matmul(R, inputs)
    points_out = points_out.permute(0, 2, 1).contiguous()
    
    if gt is not None:
        gt = gt.permute(0, 2, 1).contiguous()
        gt_out = torch.matmul(R, gt)
        gt_out = gt_out.permute(0, 2, 1).contiguous()
    else:
        gt_out = None
    return points_out, gt_out
    
    
def rotate_z_2 (inputs, gt, init_angle=0.0, random_rotate=True, descrete_rotate=None):
    batch_size = inputs.shape[0]
    if random_rotate:
        assert init_angle is None, 'init_angle should be set None when random_rotate is True'
        if descrete_rotate is None:
            # angles = np.random.uniform(0, 2*np.pi, batch_size)
            
            r = 45
            x = np.random.uniform(0, 4, batch_size)
            x = x//1*90 + x%1*r - r/2
            angles = x/180*np.pi
            
        else:
            print ('descrete_rotate')
            angles = np.random.randint(descrete_rotate, size=batch_size)/descrete_rotate*2*np.pi  # descrete_rotate=64   
    else:
        angles = init_angle * np.ones(batch_size, dtype=np.float32)
        
    R = np.array([[np.cos(angles),       -np.sin(angles),        np.zeros_like(angles)],
                  [np.sin(angles),        np.cos(angles),        np.zeros_like(angles)],
                  [np.zeros_like(angles), np.zeros_like(angles), np.ones_like(angles)]], dtype=np.float32)
    R = torch.tensor(R, device=inputs.device).permute(2, 0, 1).contiguous()
    
    inputs = inputs.permute(0, 2, 1).contiguous()
    points_out = torch.matmul(R, inputs)
    points_out = points_out.permute(0, 2, 1).contiguous()
    
    if gt is not None:
        gt = gt.permute(0, 2, 1).contiguous()
        gt_out = torch.matmul(R, gt)
        gt_out = gt_out.permute(0, 2, 1).contiguous()
    else:
        gt_out = None
    return points_out, gt_out
    
    
def batch_rotate (points_in, init_angle, res_pt=64):
    # init_angle = init_angle.cpu().numpy()
    angles = init_angle * (2*np.pi/res_pt)
    R = np.array([[np.cos(angles),       -np.sin(angles),        np.zeros_like(angles)],
                  [np.sin(angles),        np.cos(angles),        np.zeros_like(angles)],
                  [np.zeros_like(angles), np.zeros_like(angles), np.ones_like(angles)]], dtype=np.float32)
    
    R = torch.Tensor(R).cuda()
    R = R.permute([2,0,1]).contiguous()
    
    points_in = points_in.permute([0,2,1]).contiguous()
    points_out = torch.matmul(R,points_in).contiguous()
    points_out = points_out.permute([0,2,1]).contiguous()
    
    return points_out


class dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, train_phase, npoints=2048, fts_path=None, coarse_path=None):
            
        self.input_path = os.path.join(dataset_root, '%s_data.hdf5'%train_phase)
        self.gt_path = os.path.join(dataset_root, '%s_data.hdf5'%train_phase)
        
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'])
        self.input_data = self.rotate_data(self.input_data)
        input_file.close()
        
        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array(gt_file['complete_pcds'])
        self.gt_data = self.rotate_data(self.gt_data)
        gt_file.close()
        
        print('Size of dataset (%s):'%train_phase)
        print('input ' + str(self.input_data.shape))
        print('gt ' + str(self.gt_data.shape))
        
        self.fts_path = fts_path
        if fts_path is not None:
            self.fts = np.array(h5py.File(self.fts_path, 'r')['%s_g'%train_phase])
            assert self.fts.shape[0] == self.input_data.shape[0] == self.gt_data.shape[0]
            print('fts ' + str(self.fts.shape))
            
        self.coarse_path = coarse_path
        if coarse_path is not None:
            self.coarse_data = np.array(h5py.File(self.coarse_path, 'r')['pretrained_%s'%train_phase])
            assert self.coarse_data.shape[0] == self.input_data.shape[0] == self.gt_data.shape[0]
            print ('coarse points ' + str(self.coarse_data.shape))

        self.len = self.input_data.shape[0]
        
    def rotate_data (self, points_in):
        batch_size = np.shape(points_in)[0]
            
        R = np.array([[np.ones(batch_size),   np.zeros(batch_size),    np.zeros(batch_size)],
                      [np.zeros(batch_size),  np.zeros(batch_size),   -np.ones(batch_size)],
                      [np.zeros(batch_size),  np.ones(batch_size),     np.zeros(batch_size)]], dtype=np.float32)
        
        R = R.transpose((2,0,1))
        points_in = points_in.transpose((0,2,1))
        points_out = np.matmul(R,points_in)
        points_out = points_out.transpose((0,2,1))
        return points_out
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        partial = torch.from_numpy(self.input_data[index]).contiguous()
        complete = torch.from_numpy(self.gt_data[index]).contiguous()
        
        if self.fts_path is not None:
            fts_g = torch.from_numpy(self.fts[index]).contiguous()
            return partial, complete, fts_g
        
        if self.coarse_path is not None:
            coarse = torch.from_numpy(self.coarse_data[index]).contiguous()
            return partial, complete, coarse
        
        if self.fts_path is None and self.coarse_path is None:
            return partial, complete
    
    
    
def view_pcd(pc_array, color_array=None, color_map=None, single_color='blue', window='open3d', return_pcd=False):
    '''
    pc_array : (n, 3)
    color_map : (n)
    '''
    colors = np.ones_like(pc_array)
    color_dict = {'blue': [0, 0, 1], 'red': [1, 0, 0], 'green': [0, 1, 0]}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)
    if color_map is not None:
        rainbow_index = np.linspace(0, 1, np.max(color_map)+1)
        # rainbow_index = np.random.permutation(rainbow_index)
        rgb_map = cm.rainbow(rainbow_index)[:, 0:3]
        colors = np.array([rgb_map[color_map[i]] for i in range(len(color_map))])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif color_array is None:
        colors = np.tile(np.array(color_dict[single_color])[None, :], [np.shape(pc_array)[0], 1])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(color_array)
    o3d.visualization.draw_geometries([pcd], window_name=window)
    
    if return_pcd:
        return np.hstack([pc_array, colors])
    
    

