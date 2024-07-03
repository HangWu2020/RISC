#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import data_util, h5py, loss
import os, sys


def ring_dist(a, b, res_pt):
    return np.min(np.vstack([np.abs(b-a), res_pt-np.abs(b-a)]), 0)

def select_candidate(candidates, res_pt, ):
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
        sorted_index = np.argsort(-group_size)
        final_candidate = np.array(center2)[sorted_index]
            
    else:
        final_candidate = np.array(center2)[0:1]

    return center, groups, final_candidate



def sel_pose(candidates, inputs, output, res_pt=64):
    output = output.reshape(1, -1, 3).repeat(len(candidates), 1, 1)
    output_rot = data_util.batch_rotate(output, candidates, res_pt)
    
    inputs = inputs.reshape(1, -1, 3).repeat(len(candidates), 1, 1)
    
    ucd_p, ucd_t, idx2 = loss.onedir_cd(inputs, output_rot)
    
    pose_id = torch.argmin(ucd_p).item()
    pose = candidates[pose_id]
    return ucd_p, pose, pose_id



def get_new_groups(pose_id, new_groups, resolution=1):
    '''
    pose_id : int
    new_groups : (m)
    -------
    '''
    new_pose = new_groups[pose_id]
    
    if new_pose == np.max(new_groups):
        if resolution == 1:
            new_groups = np.array([new_pose-1, new_pose, new_pose+1, new_pose+2, new_pose+3])
        else:
            resolution = 0.33*resolution
            new_groups = np.array([new_pose-2*resolution, new_pose-resolution, new_pose, new_pose+resolution, new_pose+2*resolution])
    
    elif new_pose == np.min(new_groups):
        if resolution == 1:
            new_groups = np.array([new_pose+1, new_pose, new_pose-1, new_pose-2, new_pose-3])
        else:
            resolution = 0.33*resolution
            new_groups = np.array([new_pose-2*resolution, new_pose-resolution, new_pose, new_pose+resolution, new_pose+2*resolution])
    
    else:
        resolution = 0.33*resolution
        new_groups = np.array([new_pose-2*resolution, new_pose-resolution, new_pose, new_pose+resolution, new_pose+2*resolution])
    
    return new_groups, resolution



def feature_align(pose_indexes, res_pt, inputs, output, iter_step=6):
    center, groups, candidates = select_candidate(pose_indexes, res_pt)
    
    all_groups = []
    all_groups.append(candidates)
    
    ucd_p, pose, pose_id = sel_pose(candidates, inputs, output, res_pt)
    new_groups = groups[pose_id]
    all_groups.append(new_groups)
    
    ucd_p, pose, pose_id = sel_pose(new_groups, inputs, output, res_pt)
    new_groups, resolution = get_new_groups(pose_id, new_groups, resolution=1)
    all_groups.append(new_groups)
    
    for i in np.arange(3, iter_step):
        ucd_p, pose, pose_id = sel_pose(new_groups, inputs, output, res_pt)
        new_groups, resolution = get_new_groups(pose_id, new_groups, resolution=resolution)
        all_groups.append(new_groups)
        
    ucd_p, pose, pose_id = sel_pose(new_groups, inputs, output, res_pt)
    
    return pose
