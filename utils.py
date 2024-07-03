#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

import s2Voxel
import pcRemap
import loss
import data_util

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os, sys
import open3d as o3d


class remap_points_kernal(Function):
    '''
    Transfer point to adapt SO(2) grid (theta, phi)

    Parameters
    ----------
    points_xyz : (b,n,3)
    x_0 : float, optional

    Returns
    -------
    points_pt : (b,n,3)
    points_pt_ad : (b,n,3)
    '''
    @staticmethod
    def forward(ctx, points_xyz, x_0=0.1):
        points_pt = torch.zeros_like(points_xyz, requires_grad=False, device='cuda', dtype=torch.float32).contiguous()
        points_pt_ad = torch.zeros_like(points_xyz, requires_grad=False, device='cuda', dtype=torch.float32).contiguous()
        pcRemap.remap(points_xyz, points_pt, points_pt_ad, x_0)
        return points_pt, points_pt_ad
    
    @staticmethod
    def backward(ctx, grad_points_pt, grad_points_pt_ad):
        return None, None

remap = remap_points_kernal.apply


class s2_gridding_kernal(Function):
    '''
    Gridding points with SO(2) grids

    Parameters
    ----------
    points_pt_ad : (b,n,3)
    grid_size : (grid_theta, grid_r)

    Returns
    -------
    grid_feature : (b, grid_size_s, grid_size_s, grid_size_r, 4)

    '''
    @staticmethod
    def forward(ctx, points_pt_ad, grid_size, scale, sigma, A, origin=True, x_0=0.1, inteval_comp=0.5):
        grid_feature = torch.zeros(points_pt_ad.shape[0], grid_size[0], grid_size[0], grid_size[1], 4, 
                                   device='cuda', dtype=torch.float32).contiguous()
        s2Voxel.voxelize(points_pt_ad, grid_feature, grid_size[0], grid_size[1], scale, sigma, A, x_0, inteval_comp, origin)
        return grid_feature
    
    @staticmethod
    def backward(ctx=None, grad_grid=None):
        return None, None, None, None, None, None, None, None
    
s2_gridding = s2_gridding_kernal.apply


def feature_align(inputs, gt, res_pt=64, k=3):
    '''
    inputs : (batch_size, res_pt)
    gt : (batch_size, res_pt)

    batch_index : (batch_size, k)
    ''' 
    dist_all = []
    for i in range(0, inputs.shape[1]):
        inputs_shift = torch.cat((inputs[:,inputs.shape[1]-i:], inputs[:,:inputs.shape[1]-i]), dim=1) #(batch_size, res_pt)
        dist_all.append(torch.sum((inputs_shift-gt)**2, dim=1, keepdim=True)) #(batch_size, 1)
    dist_all = torch.cat(dist_all, dim=1) #(batch_size, res_pt)
    
    _, batch_index = torch.topk(dist_all, k, dim=1, largest=False, sorted=True)  #(batch_size, k)
    
    s = []
    for i in range(0, k):
        a = (batch_index[:,0:i+1] == torch.zeros_like(batch_index[:,0:i+1]))
        b = (batch_index[:,0:i+1] <= 2*torch.ones_like(batch_index[:,0:i+1]))
        # c = (batch_index[:,0:i+1] >= 62*torch.ones_like(batch_index[:,0:i+1]))
        c = (batch_index[:,0:i+1] >= (res_pt-2)*torch.ones_like(batch_index[:,0:i+1]))
        s.append(torch.sum(torch.sum((a|b|c),1)>0).cpu().numpy())
        
    s = np.array(s)
    return dist_all, batch_index, s


def get_gaussian_dist (x_m=0.1, y_m=0.2, f_0=1.0):
    '''
    Parameters
    ----------
    x_m : distance from center to margin 
    y_m : f(x_m)/f(x_0)
    f_0 : f(x_0)

    Returns
    -------
    A : Scale rate of distribution
    sigma : deviation
    '''
    k = np.sqrt(-2 * np.log(y_m))
    sigma = x_m/k
    A = sigma * np.sqrt(2*np.pi) * f_0
    return A, sigma


def ring_dist(a, b, res_pt):
    return np.min(np.vstack([np.abs(b-a), res_pt-np.abs(b-a)]), 0)


def select_candidate(candidates, res_pt):
    center = []
    center.append(candidates[0])
    dist = ring_dist(candidates, center[-1], res_pt)
    
    while len(center) <= len(candidates):
        if np.max(dist) > 6:
            center.append(candidates[np.argmax(dist)])
            dist_tmp = ring_dist(candidates, center[-1], res_pt)
            dist = dist*(dist<=dist_tmp) + dist_tmp*(dist>dist_tmp)
        else:
            break
    
    dist_min = np.zeros_like(candidates)  #group id for each candidate
    group_size = np.zeros_like(center)
    
    groups = []
    groups.append(np.array([]))
    dist_last = ring_dist(candidates, center[0], res_pt)
    
    for i in range(1, len(center)):
        groups.append(np.array([]))
        dist_tmp = ring_dist(candidates, center[i], res_pt)
        dist_min = dist_min*(dist_last<=dist_tmp) + i*np.ones_like(candidates)*(dist_last>dist_tmp)    
        dist_last = dist_last*(dist_last<=dist_tmp) + dist_tmp*(dist_last>dist_tmp)
    
    for i in range(len(candidates)):
        group_id = dist_min[i]
        group_size[group_id] += 1
        if np.abs(candidates[i]-center[group_id]) > 6:
            c = candidates[i]+res_pt if candidates[i]<res_pt/2 else candidates[i]-res_pt
        else:
            c = candidates[i]
        groups[group_id] = np.hstack([groups[group_id], c])
        
    center2 = [np.median(groups[i]) for i in range(len(center))]
    
    if len(center)>1:
        sorted_index = np.argsort(-group_size)[0:2]
        if group_size[sorted_index[0]] > group_size[sorted_index[1]]:
            sorted_index = sorted_index[0]
            final_candidate = np.array(center2)[sorted_index:sorted_index+1]
        else:
            final_candidate = np.array(center2)[sorted_index]
    else:
        final_candidate = np.array(center2)[0:1]

    return center, groups, final_candidate



def batch_rotate(points_in, init_angle, res_pt=64):
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


def select_candidate_cd(inputs, xyz_coarse, batch_index, res_pt=64): 
    cds = []
    b, k = batch_index.size()
    batch_index_np = batch_index.cpu().numpy()
    for i in range(k):
        init_angle = batch_index_np[:, i]
        xyz_coarse_ = batch_rotate(xyz_coarse, init_angle, res_pt)
        cd, _, _ = loss.onedir_cd(inputs, xyz_coarse_)
        cds.append(cd.unsqueeze(1))
    cds = torch.cat(cds, dim=1)
    
    init_angle_final = torch.gather(batch_index, dim=1, index=torch.argmin(cds, dim=1, keepdim=True)).squeeze(1)
    return init_angle_final


if __name__ == '__main__':
    # a = np.array([0,1,62,5,16])
    # b = np.array([1,3,1,62,1])
    # c = np.array([1])
    # print (ring_dist(a, c, 64))
    
    candidates = np.array([2,1,62,31,33,32])
    res_pt = 64
    
    center, groups, final_candidate = select_candidate(candidates, res_pt)
