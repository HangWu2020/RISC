#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import loss

def ring_feature_dist (ft_a, ft_b, res_pt):
    '''
    ft_a : (6)
    ft_b : a scalar

    Returns : (6)
    '''
    dist_1 = torch.abs(ft_b-ft_a).unsqueeze(0)
    dist_2 = (res_pt-torch.abs(ft_b-ft_a)).unsqueeze(0)
    dist_all = torch.cat([dist_1, dist_2], 0)  # [2, res_pt]
    return torch.min(dist_all, 0)[0]


def select_candidate (candidates, res_pt):
    '''
    candidates : (6)

    dist : (6)
    final_candidate : (1) or (2)
    '''
    center = []
    center.append(candidates[0])
    dist = ring_feature_dist(candidates, center[-1], res_pt)
    
    while len(center) <= len(candidates):
        if torch.max(dist) > 6:
            center.append(candidates[torch.argmax(dist)])
            dist_tmp = ring_feature_dist(candidates, center[-1], res_pt)
            dist = dist*(dist<=dist_tmp) + dist_tmp*(dist>dist_tmp)
        else:
            break
    
    center = torch.cat([i.unsqueeze(0) for i in center])
    
    dist_min = torch.zeros_like(candidates)  #group id for each candidate
    group_size = torch.zeros_like(center)
    
    groups = []
    groups.append(torch.Tensor([]).to(candidates.device))
    dist_last = ring_feature_dist(candidates, center[0], res_pt)
    
    for i in range(1, len(center)):
        groups.append(torch.Tensor([]).to(candidates.device))
        dist_tmp = ring_feature_dist(candidates, center[i], res_pt)
        dist_min = dist_min*(dist_last<=dist_tmp) + i*torch.ones_like(candidates)*(dist_last>dist_tmp)
        dist_last = dist_last*(dist_last<=dist_tmp) + dist_tmp*(dist_last>dist_tmp)
        
    for i in range(len(candidates)):
        group_id = dist_min[i].long()
        group_size[group_id] += 1
        if torch.abs(candidates[i]-center[group_id]) > 6:
            c = candidates[i]+res_pt if candidates[i]<res_pt/2 else candidates[i]-res_pt
        else:
            c = candidates[i]
        groups[group_id] = torch.cat([groups[group_id], c.unsqueeze(0)])
    
    center2 = torch.cat([torch.quantile(groups[i], q=0.5).unsqueeze(0) for i in range(len(center))], 0)
    
    if len(center)>1:
        sorted_index = torch.argsort(-group_size)[0:2]
        if group_size[sorted_index[0]] > group_size[sorted_index[1]]:
            sorted_index = sorted_index[0]
            final_candidate = center2[sorted_index:sorted_index+1]
        else:
            final_candidate = torch.cat([center2[sorted_index[j]].unsqueeze(0) for j in [0, 1]], 0)
    else:
        final_candidate = center2[0:1]
    
    return dist, final_candidate


def init_candidate(query, ref, k=6):
    '''
    inputs : (b, grid_pt)
    gt : (b, grid_pt)

    batch_index : (b, k)
    ''' 
    dist_all = []
    for i in range(0, query.shape[1]):
        query_shift = torch.cat((query[:,query.shape[1]-i:], query[:,:query.shape[1]-i]), dim=1) #(batch_size, grid_pt)
        dist_all.append(torch.sum((query_shift-ref)**2, dim=1, keepdim=True)) #(batch_size, 1)
    dist_all = torch.cat(dist_all, dim=1) #(batch_size, res_pt)
    
    _, batch_index = torch.topk(dist_all, k, dim=1, largest=False, sorted=True)
    return dist_all, batch_index


def grid_rotate (points_in, grid_angle, grid_pt=64):
    '''
    points_in : (b,n,3)
    grid_angle : (b)
    
    points_out : (b,n,3)
    '''
    angles = grid_angle * (2*np.pi/grid_pt)
    R = torch.cat([torch.cos(angles), -torch.sin(angles), torch.zeros_like(angles),
                   torch.sin(angles),  torch.cos(angles), torch.zeros_like(angles),
                   torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles)]).reshape(3,3,-1).contiguous()
    
    R = R.permute([2,0,1]).contiguous()
    
    points_in = points_in.permute([0,2,1]).contiguous()
    points_out = torch.matmul(R,points_in)
    points_out = points_out.permute([0,2,1]).contiguous()
    
    return points_out


def deter_candidate (inputs, points, batch_index, grid_pt=64):
    '''
    inputs : (1,n,3) partial point cloud
    points : (1,n,3) for rotate
    batch_index : (2)
    
    points_rot : (1,n,3)
    final_candidate : (1)
    '''
    k = batch_index.shape[0]
    inputs = inputs.repeat(k, 1, 1).contiguous()  #(2,n,3)
    points_rot = grid_rotate (points, batch_index, grid_pt)  #(2,n,3)
    cds, _ = loss.onedir_cd(inputs, points_rot)  #(2)
    
    min_index = torch.argmin(cds)
    points_rot = points_rot[min_index].unsqueeze(0).contiguous()  #(1,n,3)
    final_candidate = batch_index[min_index].unsqueeze(0).contiguous()  #(1)
    return points_rot, final_candidate, cds


def pvc (inputs, outputs, query, ref, grid_pt, k=6):
    '''
    inputs : (b, n, 3) partial point cloud
    outputs : (b, n, 3) restored point cloud
    query : (b, feature_dim)
    ref : (b, feature_dim)

    candidate_all : (b)
    points_rot_all : (b, n, 3)
    '''
    
    candidate_all = []
    points_rot_all = []
    _, batch_index = init_candidate(query, ref, k)  #(b, k)
    
    for i in range(outputs.shape[0]):
        _, candidate = select_candidate (batch_index[i], grid_pt)
        outputs_single = outputs[i:i+1]
        if candidate.shape[0] > 1:
            inputs_single = inputs[i:i+1]
            points_rot, candidate, _ = deter_candidate (inputs_single, outputs_single, candidate, grid_pt)
        else:
            points_rot = grid_rotate (outputs_single, candidate, grid_pt)
        candidate_all.append(candidate)
        points_rot_all.append(points_rot)
    
    candidate_all = torch.cat(candidate_all, 0).contiguous()
    points_rot_all = torch.cat(points_rot_all, 0).contiguous()
    
    return candidate_all, points_rot_all


if __name__ == "__main__":
    import data_util
    import h5py
    import vis_points
    
    pc_id = 30000
    
    input_file = h5py.File('../d02_data/shapenet_dataset/mvp_test_input.h5', 'r')
    input_data = np.array((input_file['incomplete_pcds'][()]))
    points = vis_points.rot_points(input_data[pc_id])
    points_batch = np.array([points], dtype=np.float32)
    points_xyz_origin = torch.tensor(points_batch).cuda().repeat(2,1,1).contiguous()
    
    gt_file = h5py.File('../d02_data/shapenet_dataset/mvp_test_gt_2048pts.h5', 'r')
    gt_data = np.array((gt_file['complete_pcds'][()]))
    gt = vis_points.rot_points(gt_data[pc_id//26])
    gt_batch = np.array([gt], dtype=np.float32)
    gt_xyz_origin = torch.tensor(gt_batch).cuda().repeat(2,1,1).contiguous()
    
    # data_util.o3d_vis_points(points)
    # data_util.o3d_vis_points(gt)
    
    query = np.zeros([2, 64], dtype=np.float32)
    query[0,0] = 5.0
    query[0,32] = 6.0
    query[1,32] = 6.0
    
    ref = np.zeros([2, 64], dtype=np.float32)
    ref[0,0] = 6.0
    ref[0,32] = 5.0
    ref[1,16] = 5.0
    
    query = torch.Tensor(query).cuda()
    ref = torch.Tensor(ref).cuda()
    
    # candidate_all, points_rot_all = pvc (points_xyz_origin, gt_xyz_origin, query, ref, 64, k=2)
    # print (candidate_all)
    
    # data_util.o3d_vis_points(points_rot_all[0].detach().cpu().numpy())
    # data_util.o3d_vis_points(points_rot_all[1].detach().cpu().numpy())
    
    # points_out = grid_rotate (points_xyz_origin, torch.Tensor(np.array([0,32])).cuda(), grid_pt=64)
    # data_util.o3d_vis_points(points_out[0].detach().cpu().numpy())
    # data_util.o3d_vis_points(points_out[1].detach().cpu().numpy())
    
    
    grid_pt = 64
    
    # candidate_all = []
    # points_rot_all = []
    # _, batch_index = init_candidate(query, ref, k=2)  #(b, k)
    
    # batch_index_cpu = batch_index[0].cpu().numpy()
    # print (batch_index_cpu)
    
    # dist, candidate = select_candidate (batch_index[0], grid_pt)   
    # print (dist, candidate)
    
    batch_index2 = torch.Tensor(np.array([[32, 33, 31, 0, 3]])).cuda().contiguous()
    dist, candidate = select_candidate (batch_index2[0], grid_pt)   
    print (dist, candidate)
    
    batch_index3 = torch.Tensor(np.array([[32, 33, 31, 0, 3]]))
    dist, candidate = select_candidate (batch_index3[0], grid_pt)   
    print (dist, candidate)
    
    
    # grid_pt = 64
    
    # candidate_all = []
    # points_rot_all = []
    # _, batch_index = init_candidate(query, ref, k=2)  #(b, k)
    
    # for i in range(gt_xyz_origin.shape[0]):
    #     _, candidate = select_candidate (batch_index[i], grid_pt)
    #     print (i)
    #     print (candidate)
    #     outputs_single = gt_xyz_origin[i:i+1]
    #     if candidate.shape[0] > 1:
    #         inputs_single = points_xyz_origin[i:i+1]
    #         points_rot, candidate, cds = deter_candidate (inputs_single, outputs_single, candidate, grid_pt)
    #         print ('cd:', cds)
    #     else:
    #         points_rot = grid_rotate (outputs_single, candidate, grid_pt)
    #     print (candidate)
    #     candidate_all.append(candidate)
    #     points_rot_all.append(points_rot)
    
    # candidate_all = torch.cat(candidate_all, 0).contiguous()
    # points_rot_all = torch.cat(points_rot_all, 0).contiguous()
    
    # print (candidate_all)
    # print (points_rot_all.shape)
    


'''
Deprecated code
'''

def ring_dist(a, b, res_pt):
    return np.min(np.vstack([np.abs(b-a), res_pt-np.abs(b-a)]), 0)


def select_candidate_old (candidates, res_pt):
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


def feature_align(inputs, gt, k=3):
    '''
    inputs : (b, grid_pt)
    gt : (b, grid_pt)

    batch_index : (b, k)
    ''' 
    dist_all = []
    for i in range(0, inputs.shape[1]):
        inputs_shift = torch.cat((inputs[:,inputs.shape[1]-i:], inputs[:,:inputs.shape[1]-i]), dim=1) #(batch_size, grid_pt)
        dist_all.append(torch.sum((inputs_shift-gt)**2, dim=1, keepdim=True)) #(batch_size, 1)
    dist_all = torch.cat(dist_all, dim=1) #(batch_size, res_pt)
    
    _, batch_index = torch.topk(dist_all, k, dim=1, largest=False, sorted=True)
    
    s = []
    for i in range(0, k):
        a = (batch_index[:,0:i+1] == torch.zeros_like(batch_index[:,0:i+1]))
        b = (batch_index[:,0:i+1] <= 2*torch.ones_like(batch_index[:,0:i+1]))
        c = (batch_index[:,0:i+1] >= 62*torch.ones_like(batch_index[:,0:i+1]))       
        s.append(torch.sum(torch.sum((a|b|c),1)>0).cpu().numpy())
        
    s = np.array(s)
    return dist_all, batch_index, s