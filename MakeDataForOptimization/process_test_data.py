# 1. slam result + local pose = global pose
# 2. save global pose result
import open3d
from utils.skeleton import Skeleton
import os
from slam_reader import SLAMReader
from scipy.io import loadmat
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from copy import deepcopy
import pickle
from natsort import natsorted


class TestDataPreprocessor:
    def __init__(self, slam_result_path, heatmap_dir, depth_dir, gt_path, start_frame, end_frame, fps,
                 mat_start_frame):
        self.heatmap_dir = heatmap_dir
        self.depth_dir = depth_dir
        self.slam_result_path = slam_result_path
        self.slam_reader = SLAMReader(fps=fps)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.skeleton_model = Skeleton(calibration_path='utils/fisheye/fisheye.calibration.json')
        self.gt_global_skeleton = self.load_gt_data(gt_path=gt_path, start_frame=start_frame, end_frame=end_frame,
                                                    mat_start_frame=mat_start_frame)
        self.local_pose_list, self.heatmap_list = self.get_local_pose(self.heatmap_dir, self.depth_dir,
                                                                      self.start_frame, self.end_frame)
        self.trajectory, self.R, self.t = self.slam_reader.read_trajectory_new(self.slam_result_path,
                                                                               self.local_pose_list,
                                                                               self.gt_global_skeleton,
                                                                               start_frame=start_frame,
                                                                               end_frame=end_frame)
        

    
    def load_gt_data(self, gt_path, start_frame, end_frame, mat_start_frame):
        with open(gt_path, 'rb') as f:
            pose_gt = pickle.load(f)
        clip = []
        for i in range(start_frame, end_frame):
            clip.append(pose_gt[i - mat_start_frame])
        
        skeleton_list = clip
        
        return skeleton_list
    
    def get_local_pose(self, heatmaps_dir, depths_dir, start_frame, end_frame):
        local_pose_list = []
        heatmap_list = []
        heatmaps_filename_list = natsorted(os.listdir(heatmaps_dir))[start_frame: end_frame]
        depths_filename_list = natsorted(os.listdir(depths_dir))[start_frame: end_frame]
        
        for heatmaps_filename, depths_filename in tqdm(zip(heatmaps_filename_list, depths_filename_list)):
            heatmap_path = os.path.join(heatmaps_dir, heatmaps_filename)
            depth_path = os.path.join(depths_dir, depths_filename)
            
            self.skeleton_model.set_skeleton_from_file(heatmap_file=heatmap_path,
                                                       depth_file=depth_path,
                                                       # bone_length_file='/home/wangjian/Develop/BodyPoseRefiner/utils/fisheye/mean3D.mat',
                                                       to_mesh=False)
            local_skeleton = self.skeleton_model.skeleton
            local_pose_list.append(local_skeleton)
            heatmap_mat = loadmat(heatmap_path)
            heatmap = heatmap_mat['heatmap']
            heatmap_list.append(heatmap)
        return local_pose_list, heatmap_list
    
    def render_body_sequence(self, visualization=False):
        global_skeleton_list = []
        for i in range(len(self.trajectory)):
            slam_coord = self.trajectory[i]
            local_skeleton = self.local_pose_list[i]
            local_skeleton_points = open3d.geometry.PointCloud()
            local_skeleton_points.points = open3d.utility.Vector3dVector(local_skeleton)
            local_skeleton_points = local_skeleton_points.transform(slam_coord)
            # local_skeleton_points = local_skeleton_points.rotate(-Rotation.from_matrix(self.R).as_euler('xyz'),
            #                                                      center=False)
            # local_skeleton_points = local_skeleton_points.translate(-self.t)
            global_skeleton = np.asarray(local_skeleton_points.points)
            global_skeleton_list.append(global_skeleton)
        # align head to zero position
        # head_position = deepcopy(global_skeleton_list[0][0])
        # for i in range(len(global_skeleton_list)):
        #     global_skeleton_list[i] -= head_position
        if visualization:
            mesh_list = []
            for skeleton in global_skeleton_list:
                self.skeleton_model.skeleton = skeleton
                self.skeleton_model.skeleton_to_mesh()
                mesh_list.append(self.skeleton_model.skeleton_mesh)
            mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            open3d.visualization.draw_geometries(mesh_list + [mesh_frame])
            
            return global_skeleton_list, mesh_list
        else:
            return global_skeleton_list, []
    
    def rotate_gt_global_skeleton(self, gt_skeleton_list, R, t):
        result_gt_skeleton_list = []
        for i in range(len(gt_skeleton_list)):
            gt_skeleton = gt_skeleton_list[i]
            gt_skeleton_points = open3d.geometry.PointCloud()
            gt_skeleton_points.points = open3d.utility.Vector3dVector(gt_skeleton)
            gt_skeleton_points = gt_skeleton_points.rotate(-Rotation.from_matrix(R).as_euler('xyz'),
                                                           center=False)
            gt_skeleton_points = gt_skeleton_points.translate(t)
            gt_skeleton = np.asarray(gt_skeleton_points.points)
            result_gt_skeleton_list.append(gt_skeleton)
        return result_gt_skeleton_list
    
    def align_estimated_and_gt_skeleton(self, estimated_skeleton_list, gt_skeleton_list):
        estimated_skeleton_head_position = estimated_skeleton_list[0][0]
        gt_head_position = gt_skeleton_list[0][0]
        
        distance = estimated_skeleton_head_position - gt_head_position
        
        distance = deepcopy(distance)
        for gt_pose in gt_skeleton_list:
            gt_pose += distance
        return gt_skeleton_list


def main(slam_result_path, heatmap_dir, depth_dir, gt_path,
         start_frame, end_frame, out_dir, fps, mat_start_frame):
    visualizer = TestDataPreprocessor(
        slam_result_path=slam_result_path,
        heatmap_dir=heatmap_dir,
        depth_dir=depth_dir,
        gt_path=gt_path,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        mat_start_frame=mat_start_frame)
    
    # out_dir = 'corrected_data/studio-lingjie1_visualization/{}'.format(out_data_id)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, 'test_data.pkl')
    
    local_estimated_skeleton_list = visualizer.local_pose_list
    heatmap_list = visualizer.heatmap_list
    slam_result_list = visualizer.trajectory
    
    estimated_skeleton_list, estimated_mesh_list = visualizer.render_body_sequence(visualization=False)
    rotated_gt_global_pose_list = visualizer.gt_global_skeleton
    
    out = {
        "gt_global_skeleton": rotated_gt_global_pose_list,
        'estimated_global_skeleton': estimated_skeleton_list,
        'estimated_local_skeleton': local_estimated_skeleton_list,
        'camera_pose_list': slam_result_list,
        "heatmap_list":heatmap_list
    }
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    
    mpjpe = []
    for i in range(len(rotated_gt_global_pose_list)):
        distance = rotated_gt_global_pose_list[i] - estimated_skeleton_list[i]
        distance = np.linalg.norm(distance, axis=1)
        mpjpe.append(np.mean(distance))
    print("The initial mpjpe is: {}".format(np.mean(mpjpe)))


if __name__ == '__main__':
    slam_result_path = r'data/studio-lingjie1/frame_trajectory.txt',
    heatmap_dir = r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\EgocentricData\REC23102020\studio-lingjie1\heatmaps',
    depth_dir = r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\EgocentricData\REC23102020\studio-lingjie1\depths',
    gt_path = 'data/studio-lingjie1/lingjie1.pkl',

    total_start_frame = 551
    total_end_frame = 3300


    test_size = 100
    for i in range(total_start_frame, total_end_frame - test_size, test_size):
        print("running test sequence from {} to {}".format(i, i + test_size))
        start_frame = i
        end_frame = i + test_size
        out_dir = 'corrected_data/studio-lingjie1_visualization/data_start_{}_end_{}'.format(start_frame, end_frame)
        main(slam_result_path, heatmap_dir, depth_dir, gt_path,
             start_frame, end_frame, out_dir, fps=25, mat_start_frame=total_start_frame)
