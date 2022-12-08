import datetime
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from copy import deepcopy
import torch
import cv2


lines = [(0, 1, 'right'), (0, 4, 'left'), (1, 2, 'right'), (2, 3, 'right'), (4, 5, 'left'), (5, 6, 'left'), (1, 7, 'right'), (4, 11, 'left'), (7, 8, 'right'), (8, 9, 'right'), (9, 10, 'right'),
                      (11, 12, 'left'), (12, 13, 'left'), (13, 14, 'left'), (7, 11, 'left')]

def draw_joints(joints, img, color=(0, 255, 0), right_color=(255, 0, 0)):
    joints_num = joints.shape[0]
    for line in lines:
        if line[0] < joints_num and line[1] < joints_num:
            start = joints[line[0]].astype(np.int)
            end = joints[line[1]].astype(np.int)
            left_or_right = line[2]
            if left_or_right == 'right':
                paint_color = right_color
            else:
                paint_color = color
            img = cv2.line(img, (start[0], start[1]), (end[0], end[1]), color=paint_color, thickness=4)
    for j in range(joints_num):
        img = cv2.circle(img, center=(joints[j][0].astype(np.int), joints[j][1].astype(np.int)),
                         radius=2, color=(0, 0, 255), thickness=6)

    return img


def trans_qrot_to_matrix(trans, rot):
    # rot is quat
    rot_mat = Rotation.from_quat(rot).as_matrix()
    mat = np.array([
        np.concatenate([rot_mat[0], [trans[0]]]),
        np.concatenate([rot_mat[1], [trans[1]]]),
        np.concatenate([rot_mat[2], [trans[2]]]),
        [0, 0, 0, 1]
    ])
    return mat

def transformation_matrix_to_translation_and_rotation(mat):
    rotation_matrix = mat[:3, :3]
    translation = mat[:3, 3]

    # rotation matrix to rotation euler
    rotation_euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
    return rotation_euler, translation


def local_skeleton_2_global_skeleton(local_pose_list, cam_2_world_mat):
    pass

def global_skeleton_2_local_skeleton(global_pose, world_2_cam_mat):
    global_pose_homo = np.concatenate([global_pose, np.ones(1)], axis=1)
    local_pose_homo = world_2_cam_mat.dot(global_pose_homo.T).T
    return local_pose_homo


def transform_pose(pose, matrix):
    pose_homo = np.concatenate([pose, np.ones(shape=(pose.shape[0], 1))], axis=1)
    new_pose_homo = matrix.dot(pose_homo.T).T
    new_pose = new_pose_homo[:, :3]
    return new_pose

def transform_pose_torch(pose, matrix):
    to_attach = torch.ones(size=(pose.shape[0], 1)).to(pose.device)
    pose_homo = torch.cat([pose, to_attach], dim=1)
    new_pose_homo = matrix.mm(pose_homo.T).T
    new_pose = new_pose_homo[:, :3]
    return new_pose

def get_concecutive_global_cam(cam_seq, last_cam):
    camera_matrix_0 = deepcopy(cam_seq[0])
    concecutive_global_cam = np.empty_like(cam_seq)
    camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
    for i, camera_pose_i in enumerate(cam_seq):
        camera_matrix_i = camera_pose_i
        camera_matrix_i_2_last = last_cam.dot(camera_matrix_0_inv.dot(camera_matrix_i))
        concecutive_global_cam[i] = camera_matrix_i_2_last
    return concecutive_global_cam

def get_relative_global_pose(local_pose_list, camera_pose_list):
    # firstly get relative camera pose list
    relative_pose_list = []
    camera_pose_0 = deepcopy(camera_pose_list[0])
    camera_matrix_0 = trans_qrot_to_matrix(camera_pose_0['loc'], camera_pose_0['rot'])
    camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
    for i, camera_pose_i in enumerate(camera_pose_list):
        camera_matrix_i = trans_qrot_to_matrix(camera_pose_i['loc'], camera_pose_i['rot'])
        camera_matrix_i_2_0 = camera_matrix_0_inv.dot(camera_matrix_i)
        local_pose = local_pose_list[i]
        transformed_local_pose = transform_pose(local_pose, camera_matrix_i_2_0)
        relative_pose_list.append(transformed_local_pose)
    return relative_pose_list

def get_relative_global_pose_with_camera_matrix(local_pose_list, camera_pose_list):
    # firstly get relative camera pose list
    relative_pose_list = []
    camera_pose_0 = deepcopy(camera_pose_list[0])
    camera_matrix_0 = camera_pose_0
    camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
    for i, camera_pose_i in enumerate(camera_pose_list):
        camera_matrix_i = camera_pose_i
        camera_matrix_i_2_0 = camera_matrix_0_inv.dot(camera_matrix_i)
        local_pose = local_pose_list[i]
        transformed_local_pose = transform_pose(local_pose, camera_matrix_i_2_0)
        relative_pose_list.append(transformed_local_pose)
    
    return np.asarray(relative_pose_list)

def get_global_pose_from_relative_global_pose(relative_global_pose_list, initial_camera_matrix):
    global_pose_list = np.zeros_like(relative_global_pose_list)
    for i, relative_global_pose in enumerate(relative_global_pose_list):
        global_pose = transform_pose(relative_global_pose, initial_camera_matrix)
        global_pose_list[i] = global_pose
    return global_pose_list

def get_relative_camera_matrix(camera_pose_1, camera_pose_2):
    camera_matrix_1_inv = torch.inverse(camera_pose_1)
    camera_matrix_2_to_1 = camera_matrix_1_inv @ camera_pose_2
    return camera_matrix_2_to_1

def get_relative_global_pose_with_camera_matrix_torch(local_pose_list, camera_pose_list):
    # firstly get relative camera pose list
    # relative_pose_list = []
    relative_pose_list = torch.zeros_like(local_pose_list)
    camera_pose_0 = camera_pose_list[0].detach().clone()
    camera_matrix_0 = camera_pose_0
    camera_matrix_0_inv = torch.inverse(camera_matrix_0)
    for i, camera_pose_i in enumerate(camera_pose_list):
        camera_matrix_i = camera_pose_i
        camera_matrix_i_2_0 = camera_matrix_0_inv.mm(camera_matrix_i)
        local_pose = local_pose_list[i]
        transformed_local_pose = transform_pose_torch(local_pose, camera_matrix_i_2_0)
        relative_pose_list[i] = transformed_local_pose
    return relative_pose_list

def get_relative_transform_from_blender(location1, rotation1, location2, rotation2):
    # input: location and rotation in blender coordinate
    # out: rotation, translation and transform matrix in OpenCV coordinate
    T_world2cv1, R_world2cv1, mat_world2cv1 = get_cv_rt_from_blender(location1, rotation1)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location2, rotation2)

    mat_cv1_2world = np.linalg.inv(mat_world2cv1)

    mat_cv1_to_cv2 = mat_cv1_2world.dot(mat_world2cv2)
    # mat cv1 to cv2 is the coordinate transformation, we need to change it to the object transformation
    mat_cv2_to_cv1 = np.linalg.inv(mat_cv1_to_cv2)

    rotation, translation = transformation_matrix_to_translation_and_rotation(mat_cv2_to_cv1)
    return rotation, translation, mat_cv2_to_cv1


def get_transform_relative_to_base_cv(base_location, base_rotation, location, rotation):
    # input: location and rotation in blender coordinate
    # out: rotation, translation and transform matrix in OpenCV coordinate
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_cv(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_cv(location, rotation)

    # mat_cv2world2 = np.linalg.inv(mat_world2cv2)
    # location_cv_homo = mat_cv2world2[:, 3]

    location_cv_homo = np.concatenate([location, np.ones(1)])

    R_cv2_2_base = R_world2cv2.T.dot(R_world2cv_base)
    new_rotation_euler = Rotation.from_matrix(R_cv2_2_base).as_euler(seq='xyz')

    new_location_homo = mat_world2cv_base.dot(location_cv_homo)
    new_location = new_location_homo[:3]

    return new_location, new_rotation_euler

def get_transform_relative_to_base_blender(base_location, base_rotation, location, rotation):
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_blender(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location, rotation)

    location_cv_homo = np.concatenate([location, np.ones(1)])

    R_cv2_2_base = R_world2cv2.T.dot(R_world2cv_base)
    new_rotation_euler = Rotation.from_matrix(R_cv2_2_base).as_euler(seq='xyz')

    new_location_homo = mat_world2cv_base.dot(location_cv_homo)
    new_location = new_location_homo[:3]

    return new_location, new_rotation_euler

# code modified from zaw lin
def get_cv_rt_from_blender(location, rotation):
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]])

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2bcam = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam.dot(location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv.dot(R_world2bcam)
    T_world2cv = R_bcam2cv.dot(T_world2bcam)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat

# code modified from zaw lin
def get_cv_rt_from_cv(location, rotation):

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2cv = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2cv = -1 * R_world2cv.dot(location)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat