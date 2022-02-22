#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from s2cnn import S2Convolution, SO3Convolution, so3_integrate
from s2cnn import s2_near_identity_grid, s2_equatorial_grid, so3_near_identity_grid, so3_equatorial_grid


class conv_3d(nn.Module):
    def __init__(self, features, bn, last_func=None):
        super().__init__()
        self.features = features
        
        # 3d convolution (Decoder)
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            conv = nn.Conv3d(nfeature_in, nfeature_out, (1, 1, 1), stride=1, padding=0)
            
            sequence.append(conv)
            sequence.append(nn.BatchNorm3d(nfeature_out)) if bn==True else None
            sequence.append(nn.ReLU())
        
        conv = nn.Conv3d(self.features[-2], self.features[-1], (1, 1, 1), stride=1, padding=0)
        sequence.append(conv)
        sequence.append(last_func) if last_func is not None else None
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x =self.model(x)
        return x



class conv_1d(nn.Module):
    def __init__(self, features, bn, last_func=None):
        super().__init__()
        self.features = features
        
        # 3d convolution (Decoder)
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            conv = nn.Conv1d(nfeature_in, nfeature_out, 1)
            
            sequence.append(conv)
            sequence.append(nn.BatchNorm1d(nfeature_out)) if bn==True else None
            sequence.append(nn.ReLU())
        
        conv = nn.Conv1d(self.features[-2], self.features[-1], 1)
        sequence.append(conv)
        sequence.append(last_func) if last_func is not None else None
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x =self.model(x)
        return x

    

class mlp(nn.Module):
    def __init__(self, features, bn, last_func=None):
        super().__init__()
        self.features = features
        
        # 3d convolution (Decoder)
        sequence = []
        for l in range(0, len(self.features)-2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l+1]
            linear = nn.Linear(nfeature_in, nfeature_out)
            
            sequence.append(linear)
            sequence.append(nn.BatchNorm1d(nfeature_out)) if bn==True else None
            sequence.append(nn.ReLU())
        
        linear = nn.Linear(self.features[-2], self.features[-1])
        sequence.append(linear)
        sequence.append(last_func) if last_func is not None else None
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        x =self.model(x)
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
            
            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True)) if bn==True else None
            #sequence.append(nn.Softplus())
            sequence.append(nn.ReLU())
            so3_grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2*b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, so3_grid))
            
        if last_relu==True:
            sequence.append(nn.ReLU())
        
        self.sequential = nn.Sequential(*sequence)
                
    def forward(self, x):
        '''
        (b,grid_r*4,grid_pt,grid_pt) -> (b,features[-1],grid_pt,grid_pt,grid_r)
        '''
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
            
            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True)) if bn==True else None
            #sequence.append(nn.Softplus())
            sequence.append(nn.ReLU())
            so3_grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2*b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, so3_grid))
            
        if last_relu==True:
            sequence.append(nn.ReLU())
        
        self.sequential = nn.Sequential(*sequence)
                
    def forward(self, x):
        '''
        (b,4,grid_pt,grid_pt,grid_r) -> (b,features[-1],grid_pt,grid_pt,grid_r)
        '''
        x_1 = self.first_layer(x)
        x_2 = self.sequential(x_1)
        return x_1, x_2