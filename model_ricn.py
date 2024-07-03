#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from s2cnn import so3_integrate
import sys
sys.path.append('./extension/pointnet2/')
import pointnet2_utils as pn2_utils

import layers
import utils
import loss
import data_util

import numpy as np
import h5py


def init_s2_feature(points_xyz, scale, sigma, A, res_pt, res_r, inteval_comp):
    batch_size = points_xyz.shape[0]
    _, points_pt_ad = utils.remap(points_xyz, 0.1)
    
    grid_feature = utils.s2_gridding(points_pt_ad, [res_pt, res_r], scale, sigma, A, True, 0.1, inteval_comp)
    grid_feature_weight = grid_feature[...,0].permute([0, 3, 1, 2]).contiguous()
    grid_feature = grid_feature.reshape(batch_size, res_pt, res_pt, res_r*4).permute([0, 3, 1, 2]).contiguous()

    return grid_feature, grid_feature_weight


def init_so3_feature(points_xyz, scale, sigma, A, res_pt, res_r):
    batch_size = points_xyz.shape[0]
    _, points_pt_ad = utils.remap(points_xyz, 0.1)
    
    grid_feature = utils.s2_gridding(points_pt_ad, [res_pt, res_r], scale, sigma, A, True, 0.1)
    grid_feature_weight = grid_feature[...,0:1].permute([0, 4, 1, 2, 3]).contiguous()
    grid_feature = grid_feature.permute([0, 4, 1, 2, 3]).contiguous()

    return grid_feature, grid_feature_weight
    
    
class Encoder_coarse(nn.Module):
    def __init__(self, enc_features):
        super().__init__()
        
        self.feature_spcnn = enc_features['encoder_l1']
        self.feature_conv3d = enc_features['encoder_conv3d']
        bandwidths = enc_features['bandwidths']
        
        bn = enc_features['bn']  # True
        last_relu = enc_features['last_relu']  # False
        last_func = enc_features['last_func']  # None
        
        self.encoder_l1 = layers.s2cnn_equators(bandwidths, self.feature_spcnn, last_relu, bn)
        self.encoder_conv3d = layers.conv_3d(self.feature_conv3d, bn, last_func)
    
    def forward(self, inputs, scale, sigma, A, res_pt, res_r, inteval_comp=0.5):
        '''
        inputs: (b, npoints, 3)

        ft_grid : (b, fts_conv3d[-1], res_pt, res_pt, res_r)
        ft_g : (b, fts_conv3d[-1])
        '''
        b, n, _ = inputs.size()
        
        _, ft = init_s2_feature(inputs, scale, sigma, A, res_pt, res_r, inteval_comp)
        _, ft_grid = self.encoder_l1(ft)
                
        b, d_grid, _, _, _ = ft_grid.size()
        
        ft_grid_max = so3_integrate(ft_grid)
        ft_grid_max_ = ft_grid_max.reshape(b, d_grid, 1, 1, 1).repeat(1, 1, res_pt, res_pt, res_pt).contiguous()
        ft_grid = torch.cat([ft_grid, ft_grid_max_], dim=1).contiguous()
        ft_grid = self.encoder_conv3d(ft_grid)
        
        ft_g = ft_grid.reshape(b, ft_grid.shape[1], -1).max(-1)[0].contiguous()  # (b, d_fts)
        
        return ft_grid, ft_g
    
    
class Decoder_coarse(nn.Module):
    def __init__(self, dec_features):
        super().__init__()
        
        features_fwd = dec_features['decoder_fwd']
        bn = dec_features['bn']
        last_func = dec_features['last_func']
        
        self.decoder_fwd = layers.mlp(features_fwd, bn, last_func)
        
    def forward(self, ft_g):
        '''
        ft_g  : (b, fts_conv3d[-1])
        
        out : (b, n, 3)
        '''
        ft_g = self.decoder_fwd(ft_g)
        out = ft_g.reshape(ft_g.shape[0], -1, 3)
        return out
    
    
# class Decoder_fine(nn.Module):
#     def __init__(self, dec_features_fine, up_rate):
#         super().__init__()
        
#         d, d_t = dec_features_fine['d'],  dec_features_fine['d_t']
#         self.up_rate = up_rate
        
#         self.transformers = nn.ModuleList([layers.transformer(d, d_t) for i in range(up_rate)])
    
#     def forward(self, xyz, k):
#         '''
#         xyz : (b, 512, 3)
#         -------
#         xyz_out : (b, 2048, 3)
#         '''
#         b, m, _ = xyz.size()
        
#         xyz_out = xyz.unsqueeze(2).repeat(1, 1, self.up_rate, 1).reshape(b, -1, 3)
        
#         xyz_det = []
#         for i in range(self.up_rate):
#             xyz_ = self.transformers[i](xyz, k)
#             xyz_det.append(xyz_)
#         xyz_det = torch.cat(xyz_det, dim=1)
        
#         xyz_out = xyz_out + xyz_det
#         return xyz_out
    
    
class RICN(nn.Module):
    def __init__(self, enc_features, dec_features, dec_features_fine, up_rate):
        super().__init__()
        
        self.encoder = Encoder_coarse(enc_features)
        self.decoder_coarse = Decoder_coarse(dec_features)
        # self.decoder_fine = Decoder_fine(dec_features_fine, up_rate)
        
    def forward(self, inputs, scale, sigma, A, res_pt, res_r, inteval_comp=0.5, output_fts=False):
        '''
        inputs: (b, npoints, 3)
        -------
        ft_grid : (b, d, res_pt, res_pt, res_r)
        ft_g : (b, d)
        xyz_coarse : (b, 1024, 3)
        '''
        ft_grid, ft_g = self.encoder(inputs, scale, sigma, A, res_pt, res_r, inteval_comp)
        xyz_coarse = self.decoder_coarse(ft_g)
        
        if output_fts:
            return xyz_coarse, ft_grid, ft_g
        else:
            return xyz_coarse
        


class Point_trans(nn.Module):
    def __init__(self, m, k, dims, d_t=64, k_t=16, bn=False, trans_bn=False, last_func=None):
        super().__init__()
        self.m, self.k, self.k_t = m, k, k_t
        
        self.net = layers.conv_2d(dims, bn, last_func)
        self.trans = layers.Transformer(dims[-1], d_t, use_bn=trans_bn)
        
    def forward(self, xyz, fts):
        '''
        xyz : (b, n, 3)
        fts : (b, d, n) or None
        -------
        center_xyz : (b, m, 3)
        center_fts : (b, d_out, m)
        '''
        ids = pn2_utils.furthest_point_sample(xyz, self.m).unsqueeze(2).repeat(1, 1, 3).long()
        center_xyz = torch.gather(xyz, dim=1, index=ids).contiguous()
        
        knn_xyz, knn_ids = layers.knn(center_xyz, self.k, xyz)
        knn_xyz = knn_xyz - center_xyz.unsqueeze(2)
        knn_xyz = knn_xyz.permute(0, 3, 1, 2).contiguous()
        
        if fts is not None:
            knn_fts = pn2_utils.grouping_operation(fts, knn_ids.int())
            knn_fts = torch.cat([knn_fts, knn_xyz], dim=1)
        else:
            knn_fts = knn_xyz
        
        knn_fts = self.net(knn_fts)
        center_fts, _ = torch.max(knn_fts, dim=3)
        
        center_fts, _ = self.trans(center_xyz, center_fts, self.k_t)
        return center_xyz, center_fts
    
    
class Upsample_unit(nn.Module):
    def __init__(self, dg, d, d_t=64, k_t=16, up_rate=4, trans_bn=False, use_softmax=True, merge_inputs=False, merge_pts=None, with_global=True, displace=True):
        super().__init__()
        
        self.k_t, self.up_rate, self.merge_inputs, self.merge_pts, self.with_global, self.displace = k_t, up_rate, merge_inputs, merge_pts, with_global, displace
        
        self.upsampler = layers.Group_transformer(d, d_t, k_t, up_rate, trans_bn, use_softmax)
        
        if self.displace:  # current xyz is complete
            self.cur_encoder_l1 = layers.conv_1d([3, 64, 128], bn=False)
            self.cur_encoder_l2 = layers.conv_1d([256, 256, d], bn=False)
            self.cur_encoder_l3 = layers.Group_transformer(d, d_t, k_t, 1, trans_bn, use_softmax)
        
        if self.with_global:
            self.out_1 = layers.conv_1d([d+dg, d, d], bn=False, with_res=True)
            self.out_2 = layers.conv_1d([d, d, d], bn=False, with_res=True)
            self.out_3 = layers.conv_1d([d+dg, d, d], bn=False, with_res=True)
            self.out_4 = layers.conv_1d([d, d//4, 3], bn=False)
        else:
            self.out_1 = layers.conv_1d([d, 2*d, d], bn=False, with_res=True)
            self.out_2 = layers.conv_1d([d, 2*d, d], bn=False, with_res=True)
            self.out_3 = layers.conv_1d([d, d, 3], bn=False)
        
    def forward(self, inputs, xyz, global_fts, last_fts=None):
        '''
        inputs : (b, n, 3)
        xyz : (b, m, 3)
        global_fts : (b, dg, 1)
        last_fts : (b, d, m)
        -------
        out_xyz : (b, m*up_rate, 3)
        out_fts : (b, d, m*up_rate)
        '''
        b, m, _ = xyz.size()
        
        if self.merge_pts is None:
            merge_pts = m
        else:
            merge_pts = self.merge_pts
        
        if self.merge_inputs:
            assert last_fts is not None
            _, idx, dist = layers.knn(inputs, 3, xyz, return_dist=True)
            dist = torch.where(dist > 0, dist, torch.zeros_like(dist))
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            inter_fts = pn2_utils.grouping_operation(last_fts, idx.int())
            inter_fts = torch.sum(inter_fts*weight.unsqueeze(1), dim=-1)
        
            xyz = torch.cat([xyz, inputs], dim=1)
            last_fts = torch.cat([last_fts, inter_fts], dim=2)
            fps_ids = pn2_utils.furthest_point_sample(xyz, merge_pts).long()
            xyz = torch.gather(xyz, dim=1, index=fps_ids.unsqueeze(2).repeat(1, 1, 3))
            last_fts = torch.gather(last_fts, dim=2, index=fps_ids.unsqueeze(1).repeat(1, last_fts.shape[1], 1))
        
        if self.displace:
            cur_fts = self.cur_encoder_l1(xyz.permute(0, 2, 1).contiguous())
            cur_fts_ = torch.max(cur_fts, dim=2, keepdim=True)[0].repeat(1, 1, merge_pts)
            cur_fts = torch.cat([cur_fts, cur_fts_], dim=1)
            cur_fts = self.cur_encoder_l2(cur_fts)
            cur_fts = self.cur_encoder_l3(last_fts, cur_fts, xyz)
        else:
            cur_fts = last_fts
            
        out_fts = self.upsampler(last_fts, cur_fts, xyz)
        
        if self.with_global:
            out_fts = torch.cat([out_fts, global_fts.repeat(1, 1, out_fts.shape[2])], dim=1)
            out_fts = self.out_2(self.out_1(out_fts))
            out_fts = torch.cat([out_fts, global_fts.repeat(1, 1, out_fts.shape[2])], dim=1)
            out_fts = self.out_3(out_fts)
            out_xyz = self.out_4(out_fts).permute(0, 2, 1).contiguous()
        else:
            out_fts = self.out_2(self.out_1(out_fts))
            out_xyz = self.out_3(out_fts).permute(0, 2, 1).contiguous()
        
        if self.displace:
            out_xyz = out_xyz + xyz.unsqueeze(1).repeat(1, self.up_rate, 1, 1).reshape(b, -1, 3)
        return out_xyz, out_fts
    
    
class Decoder_fine(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pn_partial_1 = Point_trans(m=512, k=16, dims=[3, 64, 128], d_t=64, k_t=16)
        self.pn_partial_2 = Point_trans(m=128, k=16, dims=[131, 128, 256], d_t=64, k_t=16)
        self.pn_partial_3 = layers.conv_1d([256, 512, 512], bn=False)
        
        self.upsampler_1 = Upsample_unit(dg=512, d=256, d_t=64, k_t=20, up_rate=2, 
                                         trans_bn=True, use_softmax=False, merge_inputs=False, merge_pts=None, with_global=True, displace=False)
        self.upsampler_2 = Upsample_unit(dg=512, d=256, d_t=64, k_t=20, up_rate=3, 
                                         trans_bn=True, use_softmax=True, merge_inputs=True, merge_pts=None, with_global=False, displace=True)
        self.upsampler_3 = Upsample_unit(dg=512, d=256, d_t=64, k_t=20, up_rate=2,
                                         trans_bn=True, use_softmax=True, merge_inputs=True, merge_pts=1024, with_global=False, displace=True)
    
    def forward(self, inputs, coarse=None, k=4):
        '''
        inputs : (b, n, 3)
        coarse : (b, m, 3)
        -------
        coarse_xyz : (b, 512, 3)
        refine_xyz : (b, 2048, 3)
        '''
        b, n, _ = inputs.size()
        
        center_xyz, center_fts = self.pn_partial_1(inputs, None)
        center_xyz, center_fts = self.pn_partial_2(center_xyz, center_fts)
        global_fts, _ = torch.max(self.pn_partial_3(center_fts), dim=2, keepdim=True)
        
        out_xyz_1, out_fts_1 = self.upsampler_1(inputs, center_xyz, global_fts, last_fts=center_fts)
        out_xyz_2, out_fts_2 = self.upsampler_2(inputs, out_xyz_1, global_fts, last_fts=out_fts_1)
        out_xyz_3, out_fts_3 = self.upsampler_3(inputs, out_xyz_2, global_fts, last_fts=out_fts_2)
        
        return out_xyz_1, out_xyz_2, out_xyz_3
    
    
class Inverse(nn.Module):
    def __init__(self, inv_features, new_s2=False):
        super().__init__()
        
        self.new_s2 = new_s2
        
        if new_s2:        
            self.feature_spcnn = inv_features['encoder_l1']
            self.feature_conv3d = inv_features['encoder_conv3d']
            bandwidths = inv_features['bandwidths']
            
            bn = inv_features['bn']  # True
            last_relu = inv_features['last_relu']  # False
            last_func = inv_features['last_func']  # None
            
            self.encoder_l1 = layers.s2cnn_equators(bandwidths, self.feature_spcnn, last_relu, bn)
            self.encoder_conv3d = layers.conv_3d(self.feature_conv3d, bn, last_func)
        
        self.z_conv = nn.Linear(inv_features['bandwidths'][0]*2, 1, bias=True)
        self.y_conv = nn.Conv2d(inv_features['encoder_conv3d'][-1], inv_features['encoder_conv3d'][-1],
                                (inv_features['bandwidths'][0]*2, 1), stride=1, padding=0)
        
        self.shared = layers.conv_1d(inv_features['shared'], bn=False, last_func=None, In=True)
        self.fmlp = layers.mlp(inv_features['fmlp'], bn=True, last_func=None)
        
        self.norm_mode = inv_features['norm_mode']  # 'IN', 'BN', 'Max', 'Same'
        
        if self.norm_mode == 'IN':
            self.in_part = nn.InstanceNorm1d(1, affine=False)
            self.in_full = nn.InstanceNorm1d(1, affine=False)
        if self.norm_mode == 'BN':
            self.bn_part = nn.BatchNorm1d(1)
            self.bn_full = nn.BatchNorm1d(1)
            
        
    def forward(self, ft_grid, ft_g, inputs=None, scale=None, sigma=None, A=None, res_pt=None, res_r=None, inteval_comp=0.5):
        '''
        ft_grid : (b, fts_conv3d[-1], res_pt, res_pt, res_r)
        ft_g : (b, fts_conv3d[-1])
        
        part_l2 : (b, res_pt)
        full_l2 : (b, res_pt)
        '''
        assert ft_g is not None
        
        b = ft_g.shape[0]
        
        if self.new_s2: 
            _, ft = init_s2_feature(inputs, scale, sigma, A, res_pt, res_r, inteval_comp)
            _, ft_grid = self.encoder_l1(ft)
            
            b, d_grid, _, _, _ = ft_grid.size()
            
            ft_grid_max = so3_integrate(ft_grid)
            ft_grid_max_ = ft_grid_max.reshape(b, d_grid, 1, 1, 1).repeat(1, 1, res_pt, res_pt, res_pt).contiguous()
            ft_grid = torch.cat([ft_grid, ft_grid_max_], dim=1).contiguous()
            ft_grid = self.encoder_conv3d(ft_grid)
        else:
            assert ft_grid is not None
        
        ft_p = self.z_conv(ft_grid).squeeze(-1)
        ft_p = self.y_conv(ft_p).squeeze(2)
        
        part_l2 = self.shared(ft_p)
        full_l2 = self.fmlp(ft_g).unsqueeze(1)
        
        if self.norm_mode == 'IN':
            part_l2 = self.in_part(part_l2)
            full_l2 = self.in_full(full_l2)
            part_l2 = part_l2.squeeze(1)
            full_l2 = full_l2.squeeze(1)
            
        if self.norm_mode == 'BN':
            part_l2 = self.bn_part(part_l2)
            full_l2 = self.bn_full(full_l2)
            part_l2 = part_l2.squeeze(1)
            full_l2 = full_l2.squeeze(1)
        
        if self.norm_mode == 'Max':
            part_l2 = part_l2.squeeze(1).contiguous()
            part_l2_min = torch.min(part_l2, 1, keepdim=True)[0]
            part_l2_max = torch.max(part_l2, 1, keepdim=True)[0]
            part_l2 = (part_l2 - part_l2_min)/(part_l2_max - part_l2_min)           
            
            full_l2 = full_l2.squeeze(1).contiguous()
            full_l2_min = torch.min(full_l2, 1, keepdim=True)[0]
            full_l2_max = torch.min(full_l2, 1, keepdim=True)[0]
            full_l2 = (full_l2 - full_l2_min)/(full_l2_max - full_l2_min)            
        
        if self.norm_mode == 'Same':
            part_l2 = part_l2.squeeze(1).contiguous()
            full_l2 = full_l2.squeeze(1).contiguous()
        
        loss_fts = 0.5*(loss.fts_loss(part_l2, full_l2.detach(), 'Huber') + loss.fts_loss(full_l2, part_l2.detach(), 'Huber'))        
        return part_l2, full_l2, loss_fts
        
        
