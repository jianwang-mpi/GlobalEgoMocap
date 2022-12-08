import sys

sys.path.append('./networks')
import open3d
import torch
from scipy.optimize import least_squares
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d
import numpy as np
from copy import deepcopy
from scipy.io import loadmat
import pickle
from utils.skeleton import Skeleton
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import time
from tqdm import tqdm
from utils.rigid_transform_with_scale import umeyama
from utils.utils import global_skeleton_2_local_skeleton, transform_pose
import os
from utils.utils import get_relative_global_pose_with_camera_matrix, get_relative_global_pose_with_camera_matrix_torch
from networks.models.SeqConvVAE import ConvVAE
from torch.optim import LBFGS
from utils.pytorch_gmm_from_scipy import GaussianMixturePytorchFromScipy
from scipy.ndimage import gaussian_filter1d
from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
from utils.torch_closest_rot_mat import closest_rot_mat
from calculate_errors import calculate_errors

from utils.one_euro_filter import OneEuroFilter


class BodyPoseOptimizer:
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]

    def __init__(self, camera_model_path, mean_skeleton, vae_path, seq_len, network_seq_len, latent_dim,
                 windows_size=5,
                 overlap_size=1, slide_window=False, lr=2, max_iter=25):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # read normalized bone length
        self.mean_bone_length = self.calculate_bone_length(mean_skeleton.to(self.device))
        self.mean_bone_length = torch.mean(self.mean_bone_length, 0)
        self.seq_len = seq_len
        self.network_seq_len = network_seq_len
        self.windows_size = windows_size
        self.slide_window = slide_window
        self.overlap_size = overlap_size
        self.lr = lr
        self.max_iter = max_iter

        self.initial_pose = None
        self.smoothed_pose = None
        self.initial_cam_mat = None
        self.initial_pose_2d = None
        self.weight_3d = None

        self.network = ConvVAE(in_channels=45, out_channels=45, latent_dim=latent_dim, seq_len=self.network_seq_len)
        state_dict = torch.load(vae_path)['state_dict']
        self.network.load_state_dict(state_dict)

        self.network = self.network.to(self.device)
        self.network = self.network.eval()

        self.fisheye_camera_model = FishEyeCameraCalibrated(camera_model_path)

        self.vae_weight = None
        self.gmm_weight = None
        self.smooth_weight = None
        self.bone_length_weight = None
        self.reproj_weight = None

    def set_weights(self, vae_weight, gmm_weight, smooth_weight, bone_length_weight, weight_3d, reproj_weight):
        self.vae_weight = vae_weight
        self.gmm_weight = gmm_weight
        self.smooth_weight = smooth_weight
        self.bone_length_weight = bone_length_weight
        self.weight_3d = weight_3d
        self.reproj_weight = reproj_weight

    def get_mean_bone_length(self, mean_skeleton_path):
        bone_length_mat = loadmat(mean_skeleton_path)
        mean3D = bone_length_mat['mean3D'].T  # convert shape to 15 * 3
        mean_bones_np = mean3D - mean3D[self.kinematic_parents, :]
        bone_length_np = np.linalg.norm(mean_bones_np, axis=1) / 1000
        bone_length_torch = torch.from_numpy(bone_length_np).float().to(self.device)
        return bone_length_torch

    def calculate_bone_length(self, skeleton: torch.Tensor):
        # skeleton shape: (seq_len, 15 * 3)
        skeleton = skeleton.view((-1, 15, 3))
        bone_array = skeleton - skeleton[:, self.kinematic_parents, :]
        bone_length = torch.norm(bone_array, dim=-1)
        return bone_length

    def reprojection_energy(self, x):
        x = x.view([-1, 3])
        x_2d = self.fisheye_camera_model.world2camera_pytorch(x)
        distance = x_2d - self.initial_pose_2d
        return torch.sum(torch.square(distance))

    def bilinear_interpolate_torch(self, im, x, y):
        # first y, then x
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())

        return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
            torch.t(Id) * wd)

    def reprojection_energy_heatmap(self, pose):
        pose = pose.view([-1, 3])
        result = torch.empty(size=(pose.shape[0],)).float().to(self.device)
        pose_2d = self.fisheye_camera_model.world2camera_pytorch(pose)
        for i in range(len(pose_2d)):
            x = (pose_2d[i][0] - 128) / 16
            y = pose_2d[i][1] / 16
            heat = self.bilinear_interpolate_torch(self.heatmap_seq[i], x, y)
            result[i] = heat
        return -torch.sum(result)

    def reprojection_energy_heatmap_fast(self, pose):
        pose = pose.view([-1, 3])
        result = torch.empty(size=(pose.shape[0],)).float().to(self.device)
        pose_2d = self.fisheye_camera_model.world2camera_pytorch(pose)
        pose_2d[:, 0] = pose_2d[:, 0] - 128
        pose_2d = (pose_2d - 512) / 512
        pose_2d = pose_2d.view(-1, 1, 1, 2)
        heatmap_seq = self.heatmap_seq.view(-1, 1, self.heatmap_seq.shape[-2], self.heatmap_seq.shape[-1])
        sampled_heat = torch.nn.functional.grid_sample(heatmap_seq, pose_2d, align_corners=True)

        return -torch.sum(sampled_heat)

    def cam_energy(self, cam_mat_list: torch.Tensor):
        distance = cam_mat_list - self.initial_cam_mat

        # first_distance = cam_mat_list[0] - self.initial_cam_mat[0]
        E_cam = torch.sum(torch.square(distance))
        # E_cam_first = torch.sum(torch.square(first_distance))
        return E_cam  # + 100 * E_cam_first

    def cam_constraint(self, cam_mat_list: torch.Tensor):
        rot_mat = cam_mat_list[:, :3, :3]
        translation = cam_mat_list[:, :3, 3]

        rot_det = torch.det(rot_mat)
        E_det = torch.sum(torch.square(rot_det - 1))

        rot_transpose = rot_mat.permute((0, 2, 1))
        diff_orth = rot_mat.matmul(rot_transpose) - torch.eye(3).to(cam_mat_list.device)
        E_orth = torch.sum(torch.square(diff_orth))

        return E_det + E_orth

    def bone_length_energy(self, x):
        x = x.view([self.seq_len, 15, 3])

        predicted_bone_length = self.calculate_bone_length(x)
        E_bone_length = torch.sum(torch.square(predicted_bone_length - self.mean_bone_length))
        return E_bone_length

    # def bone_length_energy(self, x):
    #     x = x.view([self.seq_len, 15, 3])
    #
    #     predicted_bone_length = self.calculate_bone_length(x)
    #     mean_bone_length = torch.mean(predicted_bone_length, 0).view(-1, 15)
    #
    #     E_bone_length = torch.sum(torch.square(predicted_bone_length - mean_bone_length))
    #     return E_bone_length

    def soft_smooth_energy(self, x):
        x = x.view([self.seq_len, 15, 3])
        distance = self.smoothed_pose - x
        E_smooth = torch.sum(torch.square(distance))
        return E_smooth

    def soft_smooth_energy_center(self, x):
        x = x.view([self.seq_len, 15, 3])
        smoothed_pose_copy = x.detach().clone().cpu().numpy()
        smoothed_pose_copy = gaussian_filter1d(smoothed_pose_copy, sigma=1, axis=0)
        distance = x - torch.from_numpy(smoothed_pose_copy).to(self.device)
        E_smooth = torch.sum(torch.square(distance[1:-1]))
        return E_smooth

    def smooth_accelerate(self, x):
        x = x.view([self.seq_len, 15, 3])
        x_0 = x[1:]
        x_1 = x[:-1]
        velocity = x_1 - x_0
        acceleration = velocity[:-1] - velocity[1:]
        return torch.sum(torch.square(acceleration))

    def pose_energy_3d(self, x):
        distance = x - self.initial_pose
        E_3d = torch.sum(torch.square(distance))
        return E_3d

    def vae_energy(self, hidden_parameter):
        # hidden parameter shape: (1, hidden_size)
        result = torch.sum(torch.square(hidden_parameter))
        return result

    def local_pose_2_global_pose(self, local_pose_list, cv_2_world_mat_list):
        local_pose_list = local_pose_list.view((self.seq_len, 15, 3))
        cv_2_world_mat_list = cv_2_world_mat_list.view((self.seq_len, 4, 4))
        relative_global_pose = get_relative_global_pose_with_camera_matrix_torch(local_pose_list, cv_2_world_mat_list)
        return relative_global_pose

    def total_loss(self, hidden_parameter):
        relative_global_pose = self.network.decode_to_bodypose(hidden_parameter)
        relative_global_pose = relative_global_pose.squeeze(0).contiguous()
        E_3d = self.pose_energy_3d(relative_global_pose)
        E_smooth = self.smooth_accelerate(relative_global_pose)
        E_bone_length = self.bone_length_energy(relative_global_pose)
        if self.reproj_weight == 0:
            E_reproj = 0
        else:
            E_reproj = self.reprojection_energy_heatmap_fast(relative_global_pose)


        E_vae = self.vae_energy(relative_global_pose)
        return self.weight_3d * E_3d + self.smooth_weight * E_smooth + self.bone_length_weight * E_bone_length \
               + self.vae_weight * E_vae + self.reproj_weight * E_reproj

    def optimize_pose_seq_pytorch_LBFGS(self, relative_global_pose, heatmap_seq, smoothed_pose):
        relative_global_pose = torch.from_numpy(relative_global_pose).float().to(self.device)
        heatmap_seq = torch.from_numpy(heatmap_seq).float().to(self.device)
        smoothed_pose = torch.from_numpy(smoothed_pose).float().to(self.device)

        self.initial_pose = relative_global_pose.detach().clone()
        self.smoothed_pose = smoothed_pose.detach().clone()
        if self.reproj_weight != 0:
            self.initial_pose_2d = self.fisheye_camera_model.world2camera_pytorch(self.initial_pose.view(-1, 3))
            heatmap_seq = heatmap_seq.permute((0, 3, 1, 2)).contiguous()
            self.heatmap_seq = heatmap_seq.view(-1, heatmap_seq.shape[-2], heatmap_seq.shape[-1])
        tol = 1e-6

        relative_global_pose = relative_global_pose.view([self.seq_len, 15 * 3])
        relative_global_pose = relative_global_pose.unsqueeze(0)
        mu, var, initial_hidden_parameter = self.network.get_latent_space(relative_global_pose)
        initial_hidden_parameter = initial_hidden_parameter.detach().clone()
        initial_hidden_parameter = torch.nn.Parameter(data=initial_hidden_parameter, requires_grad=True)

        lbfgs_optimizer = LBFGS(params=[initial_hidden_parameter], lr=self.lr, max_iter=self.max_iter,
                                tolerance_change=tol, line_search_fn='strong_wolfe')
        # print('---------------start optimize---------------')
        def closure():
            lbfgs_optimizer.zero_grad()
            total_loss = self.total_loss(initial_hidden_parameter)
            total_loss.backward()
            return total_loss

        lbfgs_optimizer.step(closure)
        # print('---------------end optimize---------------')

        final_relative_global_pose = self.network.decode_to_bodypose(initial_hidden_parameter)
        final_relative_global_pose = final_relative_global_pose.squeeze(0)
        relative_global_pose_parameter = final_relative_global_pose.cpu().detach().numpy()
        return relative_global_pose_parameter


def save_mesh(skeleton_model, skeleton_list, save_dir):
    # to mesh and save
    for i, pose in enumerate(skeleton_list):
        skeleton_mesh = skeleton_model.joints_2_mesh(pose)
        save_path = os.path.join(save_dir, "out_%04d.ply"%i)
        open3d.io.write_triangle_mesh(save_path, skeleton_mesh)


def local_seq_2_global_seq(local_pose_seq, cam_2_world_mat_seq):
    global_pose_seq = []
    for local_pose, cam_2_world_mat in zip(local_pose_seq, cam_2_world_mat_seq):
        global_pose = transform_pose(local_pose, cam_2_world_mat)
        global_pose_seq.append(global_pose)
    return global_pose_seq


def local_pose_2_relative_global_pose(local_pose_list, cv_2_world_mat_list):
    local_pose_list = local_pose_list.reshape((-1, 15, 3))
    cv_2_world_mat_list = cv_2_world_mat_list.reshape((-1, 4, 4))
    relative_global_pose = get_relative_global_pose_with_camera_matrix(local_pose_list, cv_2_world_mat_list)
    return relative_global_pose


def relative_global_pose_to_global_pose(relative_global_pose_list, cam_pose_list):
    initial_cam_pose = cam_pose_list[0]
    result = []
    for relative_global_pose in relative_global_pose_list:
        global_pose = transform_pose(relative_global_pose, initial_cam_pose)
        result.append(global_pose)
    return np.asarray(result)


def main(data_id, camera_model_path, vae_weight, gmm_weight, smoothness_weight, bone_length_weight, weight_3d,
         reproj_weight, visualization=False,
         final_smooth=False, merge=True,
         save=False, save_pose=False):
    with open('{}/test_data.pkl'.format(data_id), 'rb') as f:
        data = pickle.load(f)
        estimated_local_skeleton = data['estimated_local_skeleton']
        gt_skeleton = data['gt_global_skeleton']
        cv2world_mat_list = data['camera_pose_list']
        heatmap_list = data['heatmap_list']
        estimated_local_skeleton = np.asarray(estimated_local_skeleton)
        gt_skeleton = np.asarray(gt_skeleton)
        cv2world_mat_list = np.asarray(cv2world_mat_list)
        heatmap_list = np.asarray(heatmap_list)

    skeleton_model = Skeleton(calibration_path=None)

    seq_len = 10
    dilation_size = 1
    overlap_size = 2

    body_pose_optimizer = BodyPoseOptimizer(camera_model_path=camera_model_path,
                                            mean_skeleton=torch.from_numpy(gt_skeleton).float(),
                                            vae_path='networks/logs/real_full_dataset_latent_2048_len_10_slide_window_step_1_kl_0.5/checkpoints/19.pth.tar',
                                            latent_dim=2048,
                                            network_seq_len=seq_len,
                                            seq_len=seq_len,
                                            windows_size=dilation_size,
                                            overlap_size=overlap_size,
                                            lr=2, max_iter=25)

    local_body_pose_optimizer = BodyPoseOptimizer(camera_model_path=camera_model_path,
                                                  mean_skeleton=torch.from_numpy(gt_skeleton).float(),
                                                  vae_path='networks/logs/only_local_full_dataset_latent_2048_len_10_kl_0.5_2/checkpoints/19.pth.tar',
                                                  latent_dim=2048,
                                                  network_seq_len=seq_len,
                                                  seq_len=seq_len,
                                                  windows_size=dilation_size,
                                                  overlap_size=overlap_size,
                                                  lr=2, max_iter=25)

    body_pose_optimizer.set_weights(vae_weight=vae_weight, gmm_weight=gmm_weight, smooth_weight=smoothness_weight,
                                    bone_length_weight=0.01, weight_3d=weight_3d, reproj_weight=0)

    local_body_pose_optimizer.set_weights(vae_weight=vae_weight, gmm_weight=gmm_weight,
                                          smooth_weight=smoothness_weight / 100,
                                          bone_length_weight=bone_length_weight, weight_3d=weight_3d / 10000,
                                          reproj_weight=reproj_weight)

    final_estimated_seq = []
    final_estimated_local_seq = []
    mid_local_pose_seq = []
    mid_estimated_seq = []
    final_optimized_seq = []
    final_gt_seq = []

    time_local = []
    time_global = []

    for i in tqdm(range(0, len(estimated_local_skeleton) - seq_len + 1, seq_len - overlap_size)):
        estimated_local_seq = estimated_local_skeleton[i: i + seq_len]
        cam_seq = cv2world_mat_list[i: i + seq_len]
        gt_seq = gt_skeleton[i: i + seq_len]
        heatmap_seq = heatmap_list[i: i + seq_len]

        # optimize local seq
        estimated_local_seq = np.asarray(estimated_local_seq)
        final_estimated_local_seq.append(deepcopy(estimated_local_seq))
        heatmap_seq = np.asarray(heatmap_seq)
        gt_global_pose_seq = np.asarray(gt_seq)

        smoothed_local_skeleton = deepcopy(estimated_local_seq)
        smoothed_local_skeleton = gaussian_filter1d(smoothed_local_skeleton, sigma=1, axis=0)
        import timeit
        start_time = timeit.default_timer()
        local_pose_result = local_body_pose_optimizer.optimize_pose_seq_pytorch_LBFGS(
            estimated_local_seq,
            heatmap_seq,
            smoothed_local_skeleton)
        time_local.append(timeit.default_timer() - start_time)

        mid_local_pose_seq.append(deepcopy(local_pose_result))

        estimated_relative_global_pose_seq = get_relative_global_pose_with_camera_matrix(estimated_local_seq,
                                                                                         cam_seq)

        local_optimized_relative_global_pose_seq = get_relative_global_pose_with_camera_matrix(local_pose_result,
                                                                                               cam_seq)

        estimated_global_pose_seq = relative_global_pose_to_global_pose(estimated_relative_global_pose_seq, cam_seq)
        local_optimized_global_pose_seq = relative_global_pose_to_global_pose(local_optimized_relative_global_pose_seq,
                                                                              cam_seq)
        # gt_relative_global_pose_seq = get_relative_global_pose_with_camera_matrix(gt_seq, cam_seq)

        smoothed_relative_global_skeleton = deepcopy(local_optimized_relative_global_pose_seq)
        smoothed_relative_global_skeleton = gaussian_filter1d(smoothed_relative_global_skeleton, sigma=1, axis=0)

        final_estimated_seq.append(estimated_global_pose_seq)
        mid_estimated_seq.append(local_optimized_global_pose_seq)
        final_gt_seq.append(gt_global_pose_seq)


        start_time = timeit.default_timer()
        relative_global_pose_result = body_pose_optimizer.optimize_pose_seq_pytorch_LBFGS(
            local_optimized_relative_global_pose_seq,
            heatmap_seq,
            smoothed_relative_global_skeleton)
        relative_global_pose_result = relative_global_pose_result.reshape((-1, 15, 3))
        time_global.append(timeit.default_timer() - start_time)

        global_pose_result = relative_global_pose_to_global_pose(relative_global_pose_result, cam_seq)

        final_optimized_seq.append(global_pose_result)

    def merge_batches(global_pose_seq):
        if overlap_size == 0:
            return np.concatenate(global_pose_seq)
        result_seq = []
        result_seq.extend(global_pose_seq[0][:-overlap_size])
        for i in range(len(global_pose_seq) - 1):
            first_batch = global_pose_seq[i]
            second_batch = global_pose_seq[i + 1]
            mid_part = (first_batch[-overlap_size:] + second_batch[:overlap_size]) / 2
            result_seq.extend(mid_part)
            result_seq.extend(second_batch[overlap_size:-overlap_size])
        result_seq.extend(global_pose_seq[-1][-overlap_size:])
        return result_seq

    print('time local: {}'.format(np.average(time_local)))
    print('time global: {}'.format(np.average(time_global)))

    final_optimized_seq = merge_batches(np.asarray(final_optimized_seq))
    final_estimated_local_seq = merge_batches(np.asarray(final_estimated_local_seq))
    final_estimated_seq = merge_batches(np.asarray(final_estimated_seq))
    mid_local_pose_seq = merge_batches(np.asarray(mid_local_pose_seq))
    mid_estimated_seq = merge_batches(np.asarray(mid_estimated_seq))
    final_gt_seq = merge_batches(np.asarray(final_gt_seq))
    if final_smooth is True:
        print('final smooth')
        final_optimized_seq = gaussian_filter1d(final_optimized_seq, sigma=1, axis=0)

    if visualization is True:
        # to mesh and save
        estimated_mesh_list = []
        for pose in final_estimated_seq:
            estimated_mesh_list.append(skeleton_model.joints_2_mesh(pose))
        open3d.visualization.draw_geometries(estimated_mesh_list)

        optimized_mesh_list = []
        for pose in final_optimized_seq:
            optimized_mesh_list.append(skeleton_model.joints_2_mesh(pose))
        open3d.visualization.draw_geometries(optimized_mesh_list)

        gt_mesh_list = []
        for pose in final_gt_seq:
            gt_mesh_list.append(skeleton_model.joints_2_mesh(pose))
        open3d.visualization.draw_geometries(gt_mesh_list)

    if save_pose:
        dataset_dir, seq_name = os.path.split(data_id)
        dataset_name = os.path.split(dataset_dir)[1]

        from calculate_errors import align_skeleton, align_skeleton_size, global_align_skeleton_seq
        print(dataset_name, seq_name)
        out_dir = 'out/{}/{}'.format(dataset_name, seq_name)
        out_path = 'out/{}/{}/result_pose.pkl'.format(dataset_name, seq_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        with open(out_path, 'wb') as f:
            pickle.dump({'estimated_pose': final_estimated_seq, 'optimized_pose': final_optimized_seq,
                         'mid_optimized_pose': mid_estimated_seq,
                         'gt_pose': final_gt_seq}, f)

    if save:
        dataset_dir, seq_name = os.path.split(data_id)
        dataset_name = os.path.split(dataset_dir)[1]
        from calculate_errors import align_skeleton,align_skeleton_size, global_align_skeleton_seq
        print(dataset_name, seq_name)
        out_aligned_dir = 'out/{}/{}/optimized_global_aligned'.format(dataset_name, seq_name)
        if not os.path.isdir(out_aligned_dir):
            os.makedirs(out_aligned_dir)
        out_input_aligned_dir = 'out/{}/{}/input_global_aligned'.format(dataset_name, seq_name)
        if not os.path.isdir(out_input_aligned_dir):
            os.makedirs(out_input_aligned_dir)
        out_gt_aligned_dir = 'out/{}/{}/gt_global_aligned'.format(dataset_name, seq_name)
        if not os.path.isdir(out_gt_aligned_dir):
            os.makedirs(out_gt_aligned_dir)

        aligned_estimated_pose = global_align_skeleton_seq(final_estimated_seq, final_gt_seq)
        aligned_optimized_pose = global_align_skeleton_seq(final_optimized_seq, final_gt_seq)
        save_mesh(skeleton_model, aligned_optimized_pose, out_aligned_dir)
        save_mesh(skeleton_model, aligned_estimated_pose, out_input_aligned_dir)
        save_mesh(skeleton_model, final_gt_seq, out_gt_aligned_dir)

    errors = calculate_errors(final_estimated_seq, mid_estimated_seq, final_optimized_seq, final_gt_seq)
    return errors, final_estimated_seq, mid_local_pose_seq, final_optimized_seq, final_gt_seq


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data directory number')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--camera', required=False, type=str, default='utils/fisheye/pose_fisheye_fisheye.calibration_new.json')
    parser.add_argument('--vae', required=True, type=float, default=0.01)
    parser.add_argument('--gmm', required=True, type=float, default=0.001)
    parser.add_argument('--smooth', required=True, type=float, default=1)
    parser.add_argument('--bone_length', required=True, type=float, default=0)
    parser.add_argument('--weight_3d', required=True, type=float, default=0.01)
    parser.add_argument('--reproj_weight', required=True, type=float, default=0.0001)
    parser.add_argument('--save', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--final_smooth', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--merge', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_pose', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    print("running data: {}".format(args.data_path))
    from pprint import pprint

    res = main(args.data_path, camera_model_path=args.camera, vae_weight=args.vae, gmm_weight=args.gmm,
               smoothness_weight=args.smooth,
               bone_length_weight=args.bone_length,
               visualization=False, weight_3d=args.weight_3d, reproj_weight=args.reproj_weight,
               save=args.save, final_smooth=args.final_smooth, merge=args.merge, save_pose=args.save_pose)

    pprint(res[0])
