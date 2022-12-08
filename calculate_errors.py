import numpy as np

from utils.rigid_transform_with_scale import umeyama
from utils.skeleton import Skeleton
from copy import deepcopy


def global_align_skeleton_seq(estimated_seq, gt_seq):
    estimated_seq = deepcopy(estimated_seq)
    gt_seq = deepcopy(gt_seq)
    estimated_seq = np.asarray(estimated_seq).reshape((-1, 3))
    gt_seq = np.asarray(gt_seq).reshape((-1, 3))
    # aligned_pose_list = np.zeros_like(estimated_seq)
    # for s in range(estimated_seq.shape[0]):
    #     pose_p = estimated_seq[s]
    #     pose_gt_bs = gt_seq[s]
    c, R, t = umeyama(estimated_seq, gt_seq)
    pose_p = estimated_seq.dot(R) * c + t
    # aligned_pose_list[s] = pose_p
    
    return pose_p.reshape((-1, 15, 3))


def calculate_error(estimated_seq, gt_seq):
    estimated_seq = np.asarray(estimated_seq)
    gt_seq = np.asarray(gt_seq)
    distance = estimated_seq - gt_seq
    distance = np.linalg.norm(distance, axis=-1)
    m_distance = np.mean(distance)
    return m_distance


def calculate_slam_error(estimated_seq, gt_seq, align=False):
    # seq shape: n_seq, 15, 3
    estimated_seq = np.asarray(estimated_seq)
    gt_seq = np.asarray(gt_seq)
    estimated_root_seq = (estimated_seq[:, 7, :] + estimated_seq[:, 11, :]) / 2
    gt_root_seq = (gt_seq[:, 7, :] + gt_seq[:, 11, :]) / 2
    
    if align is True:
        c, R, t = umeyama(estimated_root_seq, gt_root_seq)
        estimated_root_seq = estimated_root_seq.dot(R) * c + t
    
    distance = estimated_root_seq - gt_root_seq
    distance = np.linalg.norm(distance, axis=1)
    m_distance = np.mean(distance)
    return m_distance

def align_skeleton_size(estimated_seq, gt_seq):
    estimated_seq = deepcopy(np.asarray(estimated_seq))
    gt_seq = deepcopy(np.asarray(gt_seq))
    aligned_pose_list = np.zeros_like(estimated_seq)
    for s in range(estimated_seq.shape[0]):
        pose_p = estimated_seq[s]
        pose_gt_bs = gt_seq[s]
        c, R, t = umeyama(pose_p, pose_gt_bs)
        pose_p = pose_p * c
        aligned_pose_list[s] = pose_p

    return aligned_pose_list

def align_skeleton(estimated_seq, gt_seq, skeleton_model=None):
    estimated_seq = deepcopy(np.asarray(estimated_seq))
    gt_seq = deepcopy(np.asarray(gt_seq))
    if skeleton_model is not None:
        for i in range(len(estimated_seq)):
            estimated_seq[i] = skeleton_model.skeleton_resize_single(
                estimated_seq[i],
                bone_length_file='utils/fisheye/mean3D.mat')
        for i in range(len(gt_seq)):
            gt_seq[i] = skeleton_model.skeleton_resize_single(
                gt_seq[i],
                bone_length_file='utils/fisheye/mean3D.mat')
    
    aligned_pose_list = np.zeros_like(estimated_seq)
    for s in range(estimated_seq.shape[0]):
        pose_p = estimated_seq[s]
        pose_gt_bs = gt_seq[s]
        c, R, t = umeyama(pose_p, pose_gt_bs)
        pose_p = pose_p.dot(R) * c + t
        aligned_pose_list[s] = pose_p
    
    return aligned_pose_list, gt_seq


def align_single_skeleton(estimated_pose, gt_pose, skeleton_model=None):
    estimated_pose = deepcopy(np.asarray(estimated_pose))
    gt_pose = deepcopy(np.asarray(gt_pose))
    if skeleton_model is not None:
        estimated_pose = skeleton_model.skeleton_resize_single(
            estimated_pose,
            bone_length_file='utils/fisheye/mean3D.mat')
        gt_pose = skeleton_model.skeleton_resize_single(
            gt_pose,
            bone_length_file='utils/fisheye/mean3D.mat')



    c, R, t = umeyama(estimated_pose, gt_pose)
    pose_p = estimated_pose.dot(R) * c + t

    return pose_p, gt_pose


def calculate_joint_error(estimated_seq, gt_seq):
    estimated_seq = np.asarray(estimated_seq)
    gt_seq = np.asarray(gt_seq)
    distance = estimated_seq - gt_seq
    distance = np.linalg.norm(distance, axis=2)
    joints_distance = np.mean(distance, axis=0)
    return joints_distance


def calculate_errors(final_estimated_seq, mid_estimated_seq, final_optimized_seq=None, final_gt_seq=None):
    skeleton_model = Skeleton(None)
    original_global_mpjpe = calculate_error(final_estimated_seq, final_gt_seq)
    mid_global_mpjpe = calculate_error(mid_estimated_seq, final_gt_seq)
    optimized_global_mpjpe = calculate_error(final_optimized_seq, final_gt_seq)
    
    original_camera_pos_error = calculate_slam_error(final_estimated_seq, final_gt_seq)
    optimized_camera_pos_error = calculate_slam_error(final_optimized_seq, final_gt_seq)
    

    
    # align the estimated result and original result
    
    aligned_estimated_seq_result = global_align_skeleton_seq(final_estimated_seq, final_gt_seq)
    aligned_estimated_mid_seq_result = global_align_skeleton_seq(mid_estimated_seq, final_gt_seq)
    aligned_optimized_seq_result = global_align_skeleton_seq(final_optimized_seq, final_gt_seq)

    original_aligned_camera_pos_error = calculate_slam_error(aligned_estimated_seq_result, final_gt_seq, align=False)
    mid_aligned_camera_pose_error = calculate_slam_error(aligned_estimated_mid_seq_result, final_gt_seq, align=False)
    optimized_aligned_camera_pos_error = calculate_slam_error(aligned_optimized_seq_result, final_gt_seq, align=False)
    
    aligned_original_seq_mpjpe = calculate_error(aligned_estimated_seq_result, final_gt_seq)
    aligned_mid_seq_mpjpe = calculate_error(aligned_estimated_mid_seq_result, final_gt_seq)
    aligned_optimized_seq_mpjpe = calculate_error(aligned_optimized_seq_result, final_gt_seq)
    
    # align the estimated result and original result
    aligned_estimated_result, final_gt_seq = align_skeleton(final_estimated_seq, final_gt_seq, None)
    aligned_mid_optimized_result, final_gt_seq = align_skeleton(mid_estimated_seq, final_gt_seq, None)
    aligned_optimized_result, final_gt_seq = align_skeleton(final_optimized_seq, final_gt_seq, None)
    
    aligned_original_mpjpe = calculate_error(aligned_estimated_result, final_gt_seq)
    aligned_mid_optimized_mpjpe = calculate_error(aligned_mid_optimized_result, final_gt_seq)
    aligned_optimized_mpjpe = calculate_error(aligned_optimized_result, final_gt_seq)
    
    # align the estimated result and original result
    aligned_estimated_result, final_gt_seq = align_skeleton(final_estimated_seq, final_gt_seq, skeleton_model)
    aligned_mid_optimized_result, final_gt_seq = align_skeleton(mid_estimated_seq, final_gt_seq, skeleton_model)
    aligned_optimized_result, final_gt_seq = align_skeleton(final_optimized_seq, final_gt_seq, skeleton_model)
    
    bone_length_aligned_original_mpjpe = calculate_error(aligned_estimated_result, final_gt_seq)
    bone_length_aligned_mid_optimized_mpjpe = calculate_error(aligned_mid_optimized_result, final_gt_seq)
    bone_length_aligned_optimized_mpjpe = calculate_error(aligned_optimized_result, final_gt_seq)
    joints_error = calculate_joint_error(aligned_optimized_result, final_gt_seq)
    
    from collections import OrderedDict
    result = OrderedDict({'original_global_mpjpe': original_global_mpjpe,
                          'mid_global_mpjpe': mid_global_mpjpe,
                          'optimized_global_mpjpe': optimized_global_mpjpe,
                          'original_camera_pos_error': original_camera_pos_error,
                          'optimized_camera_pos_error': optimized_camera_pos_error,
                          
                          'original_aligned_camera_pos_error': original_aligned_camera_pos_error,
                          'mid_aligned_camera_pose_error': mid_aligned_camera_pose_error,
                          'optimized_aligned_camera_pos_error': optimized_aligned_camera_pos_error,
                          
                          'original_aligned_global_mpjpe': aligned_original_seq_mpjpe,
                          "aligned_mid_seq_mpjpe": aligned_mid_seq_mpjpe,
                          'optimized_aligned_global_mpjpe': aligned_optimized_seq_mpjpe,
                          'aligned_original_mpjpe': aligned_original_mpjpe,
                          'aligned_mid_optimized_mpjpe': aligned_mid_optimized_mpjpe,
                          'aligned_optimized_mpjpe': aligned_optimized_mpjpe,
                          'bone_length_aligned_original_mpjpe': bone_length_aligned_original_mpjpe,
                          'bone_length_aligned_mid_optimized_mpjpe': bone_length_aligned_mid_optimized_mpjpe,
                          'bone_length_aligned_optimized_mpjpe': bone_length_aligned_optimized_mpjpe,
                          'joints_error': joints_error})
    return result
