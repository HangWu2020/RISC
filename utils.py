#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function

import s2Gridding
import pcRemap

import numpy as np

class remap_points_kernal (Function):
    '''
    Transfer point to adapt SO(2) grid (theta, phi)
    
    points_xyz : (b,n,3)
    x_0 : float, optional
    
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


class s2_gridding_kernal (Function):
    '''
    Gridding points on SO(2) grids
    
    points_pt_ad : (b,n,3)
    grid_size : (grid_theta, grid_r)
    
    grid_feature : (b, grid_size_s, grid_size_s, grid_size_r, 5) [x, y, z, weight, num_points]
    '''
    @staticmethod
    def forward(ctx, points_pt_ad, grid_size, grid_comp, scale, sigma, A, origin=True, x_0=0.1):
        grid_feature = torch.zeros(points_pt_ad.shape[0], grid_size[0], grid_size[0], grid_size[1], 5, 
                                   device='cuda', dtype=torch.float32).contiguous()
        s2Gridding.gridding(points_pt_ad, grid_feature, grid_size[0], grid_size[1], 
                            grid_comp[0], grid_comp[1], grid_comp[2], scale, sigma, A, x_0, origin)
        
        return grid_feature
    
    @staticmethod
    def backward(ctx=None, grad_grid=None):
        return None, None, None, None, None, None, None, None    
s2_gridding = s2_gridding_kernal.apply


def get_gaussian_dist (x_m=0.1, y_m=0.2, f_0=1.0):
    '''
    x_m : distance from center to margin 
    y_m : f(x_m)/f(x_0)
    f_0 : f(x_0)
    
    A : Scale rate of distribution
    sigma : deviation
    '''
    k = np.sqrt(-2 * np.log(y_m))
    sigma = x_m/k
    A = sigma * np.sqrt(2*np.pi) * f_0
    return A, sigma
