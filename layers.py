#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from s2cnn import S2Convolution, SO3Convolution, so3_integrate
from s2cnn import s2_near_identity_grid, s2_equatorial_grid, so3_near_identity_grid, so3_equatorial_grid


class conv_3d(nn.Module):
    def __init__(self, features, bn, last_func=None):
        super().__init__()
        self.features = features
        
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            conv = nn.Conv3d(nfeature_in, nfeature_out, (1, 1, 1), stride=1, padding=0)
            
            sequence.append(conv)
            if bn:
                sequence.append(nn.BatchNorm3d(nfeature_out))
            sequence.append(nn.ReLU())
        
        conv = nn.Conv3d(self.features[-2], self.features[-1], (1, 1, 1), stride=1, padding=0)
        sequence.append(conv)
        if last_func is not None:
            sequence.append(last_func)
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    

class conv_2d(nn.Module):
    def __init__(self, features, bn, last_func=None, In=False):
        super().__init__()
        self.features = features
        
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            conv = nn.Conv2d(nfeature_in, nfeature_out, (1, 1), stride=1, padding=0)
            
            sequence.append(conv)
            if bn:
                sequence.append(nn.BatchNorm2d(nfeature_out))
            if In:
                sequence.append(nn.InstanceNorm2d(nfeature_out, affine=False))
            sequence.append(nn.ReLU())
        
        conv = nn.Conv2d(self.features[-2], self.features[-1], (1, 1), stride=1, padding=0)
        sequence.append(conv)
        if last_func is not None:
            sequence.append(last_func)
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x = self.model(x)
        return x



class conv_1d(nn.Module):
    def __init__(self, features, bn, last_func=None, with_res=False, In=False):
        super().__init__()
        self.features = features
        
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            conv = nn.Conv1d(nfeature_in, nfeature_out, 1)
            
            sequence.append(conv)
            if bn:
                sequence.append(nn.BatchNorm1d(nfeature_out))
            sequence.append(nn.ReLU())
            
            if In:
                sequence.append(nn.InstanceNorm1d(nfeature_out, affine=False))
        
        conv = nn.Conv1d(self.features[-2], self.features[-1], 1)
        sequence.append(conv)
        if last_func is not None:
            sequence.append(last_func)
        
        self.model = nn.Sequential(*sequence)
        
        self.with_res = with_res
        if self.with_res:
            self.res = nn.Conv1d(features[0], features[-1], 1)
    
    def forward(self, x):
        if self.with_res:
            x = self.model(x) + self.res(x)
        else:
            x = self.model(x)
        return x

    

class mlp(nn.Module):
    def __init__(self, features, bn, last_func=None, with_res=False):
        super().__init__()
        self.features = features
        
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            linear = nn.Linear(nfeature_in, nfeature_out)
            
            sequence.append(linear)
            if bn:
                sequence.append(nn.BatchNorm1d(nfeature_out))
            sequence.append(nn.ReLU())
        
        linear = nn.Linear(self.features[-2], self.features[-1])
        sequence.append(linear)
        if last_func is not None:
            sequence.append(last_func)
        
        self.model = nn.Sequential(*sequence)
        
        self.with_res = with_res
        if self.with_res:
            self.res = nn.Linear(features[0], features[-1])
    
    def forward(self, x):
        if self.with_res:
            x = self.model(x) + self.res(x)
        else:
            x = self.model(x)
        return x
    
    

class s2cnn_equators(nn.Module):
    def __init__(self, bandwidths, features, last_relu=False, bn=False):
        super().__init__()
        assert len(bandwidths) == len(features)
        
        # S2 layer
        self.s2_grid = s2_equatorial_grid(max_beta=0, n_alpha=2*bandwidths[0], n_beta=1)
        self.first_layer = S2Convolution(features[0], features[1], bandwidths[0], bandwidths[1], self.s2_grid)
        
        sequence = []
        # SO3 layers
        for l in range(1, len(features)-1):
            nfeature_in = features[l]
            nfeature_out = features[l+1]
            b_in = bandwidths[l]
            b_out = bandwidths[l+1]
            
            if bn:
                sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.Softplus())
            #sequence.append(nn.ReLU())
            so3_grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2*b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, so3_grid))
            
        if last_relu:
            sequence.append(nn.ReLU())
        
        self.sequential = nn.Sequential(*sequence)
        
    def forward(self, x):
        x_1 = self.first_layer(x)
        x_2 = self.sequential(x_1)
        return x_1, x_2



class so3cnn_equators(nn.Module):
    def __init__(self, bandwidths, features, last_relu=False, bn=False):
        super().__init__()
        assert len(bandwidths) == len(features)
        
        # First SO3 layer
        self.so3_grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2*bandwidths[0], n_beta=1, n_gamma=1)
        self.first_layer = SO3Convolution(features[0], features[1], bandwidths[0], bandwidths[1], self.so3_grid)        
        
        sequence = []
        # SO3 layers
        for l in range(1, len(features)-1):
            nfeature_in = features[l]
            nfeature_out = features[l+1]
            b_in = bandwidths[l]
            b_out = bandwidths[l+1]
            
            if bn:
                sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            #sequence.append(nn.Softplus())
            sequence.append(nn.ReLU())
            so3_grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2*b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, so3_grid))
            
        if last_relu:
            sequence.append(nn.ReLU())
        
        self.sequential = nn.Sequential(*sequence)
                
    def forward(self, x):
        x_1 = self.first_layer(x)
        x_2 = self.sequential(x_1)
        return x_1, x_2
    
    
def distance(p1, p2):
    '''
    p1 : (b, n, c)
    p2 : (b, m, c)
    -------
    dist : (b, n, m)
    '''
    dist = torch.sum(p1**2, dim=2).unsqueeze(2) + \
           torch.sum(p2**2, dim=2).unsqueeze(1) - \
           2*torch.einsum('bnc,bmc->bnm', p1, p2)
    return dist


import sys
sys.path.append('./extension/pointnet2/')
import pointnet2_utils as pn2_utils

def knn(x1, k, x2=None, return_dist=False):

    b, m, c = x1.size()
    x2 = x1 if x2 is None else x2
    
    dist = distance(x1, x2)
    dist_k, ids = torch.topk(dist, k, largest=False, dim=-1)
    
    x2 = x2.permute(0, 2, 1).contiguous()
    x_l = pn2_utils.grouping_operation(x2, ids.int()).permute(0, 2, 3, 1).contiguous()
    if return_dist:
        return x_l, ids, dist_k
    else:
        return x_l, ids


    
class Transformer(nn.Module):
    def __init__(self, d, d_t, use_bn=False):
        super().__init__()
        
        self.map_in = nn.Conv1d(d, d_t, 1)
        self.d_t = d_t
        
        self.q = nn.Conv1d(d_t, d_t, 1)
        self.k = nn.Conv1d(d_t, d_t, 1)
        self.v = nn.Conv1d(d_t, d_t, 1)
        
        if use_bn:
            self.pos = nn.Sequential(
                nn.Conv2d(3, d_t, 1), nn.BatchNorm2d(d_t), nn.ReLU(),
                nn.Conv2d(d_t, d_t, 1)
                )
        else:
            self.pos = nn.Sequential(
                nn.Conv2d(3, d_t, 1), nn.ReLU(),
                nn.Conv2d(d_t, d_t, 1)
                )
        
        if use_bn:
            self.attn = nn.Sequential(
                nn.Conv2d(d_t, 4*d_t, 1), nn.BatchNorm2d(4*d_t), nn.ReLU(),
                nn.Conv2d(4*d_t, d_t, 1)
                )            
        else:
            self.attn = nn.Sequential(
                nn.Conv2d(d_t, 4*d_t, 1), nn.ReLU(),
                nn.Conv2d(4*d_t, d_t, 1)
                )
        
        self.map_out = nn.Conv1d(d_t, d, 1)
        
    def forward(self, xyz, fts, k):
    
        b, d, n = fts.size()
        
        xyz_l, ids = knn(xyz, k)
        
        res = fts
        
        fts_t = self.map_in(fts)
        fts_q, fts_k, fts_v = self.q(fts_t), self.k(fts_t), self.v(fts_t)
        
        fts_kl = pn2_utils.grouping_operation(fts_k, ids.int())
        fts_vl = pn2_utils.grouping_operation(fts_v, ids.int())
        
        pos = (xyz_l-xyz.unsqueeze(2)).permute(0, 3, 1, 2).contiguous()
        pos = self.pos(pos)
        
        attn = self.attn(fts_q.unsqueeze(3) - fts_kl + pos)
        attn = F.softmax(attn/np.sqrt(self.d_t), dim=-1)
        
        fts_out = torch.sum(attn*(fts_vl + pos), dim=-1)
        fts_out = self.map_out(fts_out) + res
        return fts_out, attn
    
    
class Group_transformer(nn.Module):
    def __init__(self, d, d_t=64, k_t=16, up_rate=4, use_bn=False, use_softmax=True):
        super().__init__()
        
        self.d_t, self.k_t, self.up_rate, self.use_softmax = d_t, k_t, up_rate, use_softmax
        
        self.map_v = conv_1d([2*d, d, d], bn=False, last_func=None, with_res=True)
        
        self.q, self.k, self.v = nn.Conv1d(d, d_t, 1), nn.Conv1d(d, d_t, 1), nn.Conv1d(d, d_t, 1)
        
        self.pos = conv_2d([3, d_t, d_t], bn=use_bn)
        self.attn = nn.ModuleList([conv_2d([d_t, 4*d_t, d_t], bn=use_bn) for i in range(self.up_rate)])
        
        self.map_out = nn.ModuleList([nn.Conv1d(d_t, d, 1) for i in range(self.up_rate)])
        self.res = nn.ModuleList([nn.Conv1d(d, d, 1) for i in range(self.up_rate)])
        
        
    def forward(self, fts_q, fts_k, xyz):

        b, d, m = fts_q.size()
        
        fts_v = torch.cat([fts_q, fts_k], dim=1)
        fts_v = self.map_v(fts_v)
        resi = fts_v
        
        fts_q, fts_k, fts_v = self.q(fts_q), self.k(fts_k), self.v(fts_v)
        
        xyz_l, ids = knn(xyz, self.k_t)
        fts_kl = pn2_utils.grouping_operation(fts_k, ids.int())
        fts_vl = pn2_utils.grouping_operation(fts_v, ids.int())
        
        pos = (xyz_l-xyz.unsqueeze(2)).permute(0, 3, 1, 2).contiguous()
        pos = self.pos(pos)
        
        fts_out = []
        for i in range(self.up_rate):
            attn = self.attn[i](fts_q.unsqueeze(3) - fts_kl + pos)
            if self.use_softmax:
                attn = F.softmax(attn/np.sqrt(self.d_t), dim=-1)
            fts_ = torch.sum(attn*(fts_vl + pos), dim=-1)
            fts_ = self.map_out[i](fts_) + self.res[i](resi)
            fts_out.append(fts_)
        
        fts_out = torch.cat(fts_out, dim=2)
        return fts_out
