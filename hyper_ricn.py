#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils

batch_size = 8

grid_pt = 64
grid_r = 32

enc_features = {'bandwidths' : [grid_pt//2, grid_pt//2, grid_pt//2],
                'encoder_l1': [grid_r, 64, 128],
                'encoder_conv3d': [256, 256, 512],
                'encoder_conv1d': [512, 64, 1],
                'bn': False,
                'last_relu': False,
                'last_func': None,
                }


dec_features = {'decoder_fwd': [512, 2048, 2048, 3072],
                'bn': False,
                'last_func': None
                }


dec_features_fine = {'d': 64,
                     'd_t': 128,
                     }


lr_points = 5e-4
num_epoch = 60
decay_step = 15
decay_rate = 0.3

scale = 0.501
k = 6
A, sigma = utils.get_gaussian_dist(x_m=0.1, y_m=0.5, f_0=1.0)
inteval_comp_points = 0.9

inv_features = {'bandwidths' : [grid_pt//2, grid_pt//2, grid_pt//2],
                'encoder_l1': [grid_r, 64, 128],
                'encoder_conv3d': [256, 256, 512],
                'encoder_conv1d': [512, 64, 1],
                'bn': False,
                'last_relu': False,
                'last_func': None,
                
                'norm_mode': 'IN',
                'fmlp': [512, 512, 256, 256, grid_pt],
                'shared': [512, 256, 128, 64, 1]
                }


batch_size_refine = 32
lr_refine = 1e-3
num_epoch_refine = 80
decay_step_refine  = 15
decay_rate_refine = 0.3


batch_size_fts = 4
lr_fts = 5e-4
num_epoch_fts = 20
decay_step_fts  = 10
decay_rate_fts = 0.5
