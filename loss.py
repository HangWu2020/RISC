#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('./extension/emd/')
sys.path.append('./extension/ChamferDistancePytorch/chamfer3D/')
import emd_module as emd
import dist_chamfer_3D as chamfer_3D

def fscore(dist1, dist2, threshold=0.0009):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


def calc_emd(output, gt, eps=0.005, iterations=100):
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out


def calc_cd(output, gt, calc_f1=False, return_raw=False, separate=False):
    '''
    output: (b, m, 3)
    gt: (b, n, 3)
    -------
    res: [cd_p: (b), cd_t: (b)]
    res-calc_f1: [..., f1: (b)]
    res-separate: [dist_p: (2, b), dist_t: (2, b), ...]
    res-return_raw: [..., dist1: (b, m), dist2: (b, n), idx1: (b, m), idx2: (b, n)]
    '''
    cham_loss = chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    
    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
        
    return res


def onedir_cd(x, gt):
    cham_loss = chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, x)
    ucd_p = torch.sqrt(dist2).mean(1)
    ucd_t = dist2.mean(1)
    return ucd_p, ucd_t, idx2


def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    '''
    x: (b, n, 3)
    gt: (b, n, 3)
    -------
    res: [loss, cd_p, cd_t]
    '''
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def fts_loss (pred_ft, ft, loss_type='L1', reduction='mean'):
    '''
    Parameters
    ----------
    pred_ft : (b, feature_dim[-1])
    ft : (b, feature_dim[-1])

    Returns
    -------
    loss : scalar
    '''
    assert loss_type in ['L1', 'L2', 'Huber']
    
    if loss_type == 'L1':
        criterion = nn.L1Loss(reduction=reduction)
        loss = criterion(pred_ft, ft)
    
    if loss_type == 'L2':
        criterion = nn.MSELoss(reduction=reduction)
        loss = criterion(pred_ft, ft)
        
    if loss_type == 'Huber':
        criterion = nn.SmoothL1Loss(reduction=reduction)
        loss = criterion(pred_ft, ft)
    
    return loss