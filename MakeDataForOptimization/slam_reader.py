# read slam results
import open3d
from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation
from utils.rigid_transform_with_scale import umeyama
from copy import deepcopy
from utils.skeleton import Skeleton


class SLAMReader:
    def __init__(self, fps=30):
        self.fps = fps
        self.skeleton_model = Skeleton(calibration_path=None)
        pass
    
    def trans_qrot_to_matrix(self, trans, rot):
        # rot is quat
        rot_mat = Rotation.from_quat(rot).as_matrix()
        mat = np.array([
            np.concatenate([rot_mat[0], [trans[0]]]),
            np.concatenate([rot_mat[1], [trans[1]]]),
            np.concatenate([rot_mat[2], [trans[2]]]),
            [0, 0, 0, 1]
        ])
        return mat
    
    def trans_rotmat_to_matrix(self, trans, rot):
        # rot is matrix
        mat = np.array([
            np.concatenate([rot[0], [trans[0]]]),
            np.concatenate([rot[1], [trans[1]]]),
            np.concatenate([rot[2], [trans[2]]]),
            [0, 0, 0, 1]
        ])
        return mat
    
    def visualize_traj(self, slam_traj_mat, gt_traj_mat):
        slam_point_cloud = open3d.geometry.PointCloud()
        slam_point_cloud.points = open3d.utility.Vector3dVector(slam_traj_mat)
        slam_point_cloud.paint_uniform_color([1, 0, 0])
        gt_pointcloud = open3d.geometry.PointCloud()
        gt_pointcloud.points = open3d.utility.Vector3dVector(gt_traj_mat)
        gt_pointcloud.paint_uniform_color([0, 1, 0])
        
        # mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
        
        open3d.visualization.draw_geometries([gt_pointcloud, slam_point_cloud])
    
    def read_trajectory_new(self, trajectory_path, local_pose_list, gt_global_pose, start_frame, end_frame):
        with open(trajectory_path, 'r') as f:
            frame_lines = f.readlines()
        
        # mat_trajectory = OrderedDict()
        # trans_and_rot = OrderedDict()
        mat_trajectory = []
        trans_and_rot = []
        for frame in frame_lines:
            frame_info = frame.strip().split()
            frame_id = round(float(frame_info[0]) * self.fps)
            trans = np.array(frame_info[1:4]).astype(np.float)
            rot = np.array(frame_info[4:]).astype(np.float)
            if start_frame <= frame_id < end_frame:
                trans_and_rot.append({"loc": trans, 'rot': rot})
        
        # trans and rot to relative trans and rot
        
        trans_and_rot = self.get_relative_camera_pose_list(trans_and_rot)
        
        gt_global_skeleton_list = np.asarray(gt_global_pose)


        # also align bone length
        # for i in range(len(gt_global_pose)):
        #     local_pose = local_pose_list[i]
        #     global_pose = gt_global_pose[i]
        #     c_p, R_p, t_p = umeyama(local_pose, global_pose)
        #     local_pose_list[i] *= c_p

        
        gt_head_locs = gt_global_skeleton_list[:, 0]

        
        slam_traj_list = []
        gt_traj_list = []
        for i in range(len(trans_and_rot)):
            slam_i_trans = trans_and_rot[i]['loc']
            slam_i_rot = trans_and_rot[i]['rot']

            # get head locs from SLAM
            slam_coord = self.trans_qrot_to_matrix(slam_i_trans, slam_i_rot)
            local_skeleton = local_pose_list[i]
            local_skeleton_points = open3d.geometry.PointCloud()
            local_skeleton_points.points = open3d.utility.Vector3dVector(local_skeleton)
            local_skeleton_points = local_skeleton_points.transform(slam_coord)
            global_skeleton = np.asarray(local_skeleton_points.points)

            slam_traj_list.append(global_skeleton[0])
            gt_traj_list.append(gt_head_locs[i])
        slam_traj_mat = np.vstack(slam_traj_list)
        gt_traj_mat = np.vstack(gt_traj_list)
        
        c, R, t = umeyama(slam_traj_mat, gt_traj_mat)
        c_1, R_1, t_1 = umeyama(gt_traj_mat, slam_traj_mat)
        
        # self.visualize_traj(slam_traj_mat.dot(R) * c + t, gt_traj_mat)
        
        # self.show_global_pose_sequence_and_slam_trajectory(global_skeleton_list, slam_traj_mat)
        
        for i in range(len(trans_and_rot)):
            trans = trans_and_rot[i]['loc']
            rot = trans_and_rot[i]['rot']
            trans = trans * c
            
            transform_mat = self.trans_qrot_to_matrix(trans, rot)
            
            mat_trajectory.append(transform_mat)


        
        return mat_trajectory, R_1, t_1
    
    def show_global_pose_sequence_and_slam_trajectory(self, global_skeleton_list, slam_traj_mat):
        global_mesh = []
        for skeleton in global_skeleton_list:
            global_mesh.append(self.skeleton_model.joints_2_mesh(skeleton))
        slam_point_cloud = open3d.geometry.PointCloud()
        slam_point_cloud.points = open3d.utility.Vector3dVector(slam_traj_mat)
        slam_point_cloud.paint_uniform_color([1, 0, 0])
        
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
        
        open3d.visualization.draw_geometries([slam_point_cloud, mesh_frame] + global_mesh)
    
    def trans_qrot_to_matrix(self, trans, rot):
        # rot is quat
        rot_mat = Rotation.from_quat(rot).as_matrix()
        mat = np.array([
            np.concatenate([rot_mat[0], [trans[0]]]),
            np.concatenate([rot_mat[1], [trans[1]]]),
            np.concatenate([rot_mat[2], [trans[2]]]),
            [0, 0, 0, 1]
        ])
        return mat
    
    def matrix_to_trans_qrot(self, matrix):
        translation = np.asarray([matrix[0][3], matrix[1][3], matrix[2][3]])
        
        rotation_matrix = np.asarray([matrix[0][:3], matrix[1][:3], matrix[2][:3]])
        
        qrot = Rotation.from_matrix(rotation_matrix).as_quat()
        
        return translation, qrot
    
    def get_relative_camera_pose_list(self, camera_pose_list):
        # firstly get relative camera pose list
        relative_camera_pose_list = []
        camera_pose_0 = deepcopy(camera_pose_list[0])
        camera_matrix_0 = self.trans_qrot_to_matrix(camera_pose_0['loc'], camera_pose_0['rot'])
        camera_matrix_0_inv = np.linalg.inv(camera_matrix_0)
        for i, camera_pose_i in enumerate(camera_pose_list):
            camera_matrix_i = self.trans_qrot_to_matrix(camera_pose_i['loc'], camera_pose_i['rot'])
            camera_matrix_i_to_0 = camera_matrix_0_inv.dot(camera_matrix_i)
            trans, qrot = self.matrix_to_trans_qrot(camera_matrix_i_to_0)
            relative_camera_pose_list.append({'loc': trans,
                                              'rot': qrot})
        return relative_camera_pose_list
    
    def read_trajectory(self, trajectory_path, start_frame, end_frame, scale=1):
        with open(trajectory_path, 'r') as f:
            frame_lines = f.readlines()

        # mat_trajectory = OrderedDict()
        # trans_and_rot = OrderedDict()
        mat_trajectory = []
        trans_and_rot = []
        for frame in frame_lines:
            frame_info = frame.strip().split()
            frame_id = round(float(frame_info[0]) * self.fps)
            trans = np.array(frame_info[1:4]).astype(np.float)
            rot = np.array(frame_info[4:]).astype(np.float)
            if start_frame <= frame_id < end_frame:
                trans_and_rot.append({"loc": trans, 'rot': rot})

        # trans and rot to relative trans and rot

        trans_and_rot = self.get_relative_camera_pose_list(trans_and_rot)


        slam_traj_list = []
        for i in range(len(trans_and_rot)):
            trans = trans_and_rot[i]['loc']
            rot = trans_and_rot[i]['rot']
            trans = trans * scale

            transform_mat = self.trans_qrot_to_matrix(trans, rot)

            mat_trajectory.append(transform_mat)

        return mat_trajectory
