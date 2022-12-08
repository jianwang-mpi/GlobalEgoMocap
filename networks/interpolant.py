import open3d
import sys

sys.path.append('..')
import torch
from models.SeqConvVAE import ConvVAE
import pickle
from utils.skeleton import Skeleton
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from old_dataset import AMASSDataset
from utils.utils import transform_pose
from utils.rigid_transform_with_scale import umeyama


def calculate_error(estimated_seq, gt_seq):
    estimated_seq = np.asarray(estimated_seq)
    gt_seq = np.asarray(gt_seq)
    distance = estimated_seq - gt_seq
    distance = np.linalg.norm(distance, axis=2)
    m_distance = np.mean(distance)
    return m_distance


def global_align_skeleton_seq(estimated_seq, gt_seq):
    estimated_seq = np.asarray(estimated_seq).reshape((-1, 3))
    gt_seq = np.asarray(gt_seq).reshape(-1, 3)
    # aligned_pose_list = np.zeros_like(estimated_seq)
    # for s in range(estimated_seq.shape[0]):
    #     pose_p = estimated_seq[s]
    #     pose_gt_bs = gt_seq[s]
    c, R, t = umeyama(estimated_seq, gt_seq)
    pose_p = estimated_seq.dot(R) * c + t
    # aligned_pose_list[s] = pose_p
    
    return pose_p.reshape((-1, 15, 3))


def align_skeleton(estimated_seq, gt_seq):
    estimated_seq = np.asarray(estimated_seq)
    gt_seq = np.asarray(gt_seq)
    aligned_pose_list = np.zeros_like(estimated_seq)
    for s in range(estimated_seq.shape[0]):
        pose_p = estimated_seq[s]
        pose_gt_bs = gt_seq[s]
        c, R, t = umeyama(pose_p, pose_gt_bs)
        pose_p = pose_p.dot(R) * c + t
        aligned_pose_list[s] = pose_p
    
    return aligned_pose_list

class InterpolateDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        data_i = self.data[i]
        data_i = data_i.reshape((-1, 45))
        data_i = torch.from_numpy(data_i).float()
        return data_i

class Interpolate:
    def __init__(self, pkl_path, network_path, out_path, seq_length=20, windows_size=5, latent_dim=128):
        self.seq_length = seq_length
        self.out_path = out_path
        self.network = ConvVAE(in_channels=45, out_channels=45, latent_dim=latent_dim, seq_len=self.seq_length)
        state_dict = torch.load(network_path)['state_dict']
        self.network.load_state_dict(state_dict)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.network = self.network.to(self.device)
        
        self.skeleton_model = Skeleton(None)
        
        self.test_dataset = InterpolateDataset(pkl_path=pkl_path)
    
    def save_sample(self, sample, out_path):
        skeleton_list = self.skeleton_model.joint_list_2_mesh_list(sample)
        for i, skeleton in enumerate(skeleton_list):
            open3d.io.write_triangle_mesh(os.path.join(out_path, 'out_%04d.ply' % i), skeleton)
    
    def show_example(self, sample):
        skeleton_list = self.skeleton_model.joint_list_2_mesh_list(sample)
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        open3d.visualization.draw_geometries(skeleton_list + [coor])
    
    
    def test(self, i, j, visualization=False, save=False):
        print('---------------------Start Eval-----------------------')
        self.network.eval()
        with torch.no_grad():
            seq_i = self.test_dataset[i].unsqueeze(0).to(self.device)
            seq_j = self.test_dataset[j].unsqueeze(0).to(self.device)
            
            mu_i, std_i, z_i = self.network.get_latent_space(seq_i)
            mu_j, std_j, z_j = self.network.get_latent_space(seq_j)

            first_sampled_seq = self.network.decode(z_i).permute((0, 2, 1))[0]
            first_sampled_seq = first_sampled_seq.cpu().detach_().numpy()
            first_sampled_seq = first_sampled_seq.reshape((self.seq_length, 15, 3))

            second_sampled_seq = self.network.decode(z_j).permute((0, 2, 1))[0]
            second_sampled_seq = second_sampled_seq.cpu().detach_().numpy()
            second_sampled_seq = second_sampled_seq.reshape((self.seq_length, 15, 3))

            if visualization is True:
                self.show_example(first_sampled_seq)
                self.show_example(second_sampled_seq)
            if save is True:
                for i in range(6):
                    out_dir = os.path.join(self.out_path, str(i))
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                self.save_sample(first_sampled_seq, os.path.join(self.out_path, "0"))
                self.save_sample(second_sampled_seq, os.path.join(self.out_path, "5"))

            first_z = z_i[0].cpu().detach_().numpy()
            second_z = z_j[0].cpu().detach_().numpy()

            interpolant_list = [first_z + (i / 5.) * (second_z - first_z) for i in range(1, 5)]
            interpolant_list = torch.from_numpy(np.asarray(interpolant_list)).float()
            interpolant_list = interpolant_list.to(self.device)

            sampled_result = self.network.decode(interpolant_list)
            sampled_result = sampled_result.permute((0, 2, 1))
            sampled_result = sampled_result.cpu().detach_().numpy()
            sampled_result = sampled_result.reshape((-1, self.seq_length, 15, 3))
            for i, sample in enumerate(sampled_result):
                if visualization is True:
                    self.show_example(sample)
                if save is True:
                    self.save_sample(sample, os.path.join(self.out_path, str(i + 1)))


if __name__ == '__main__':
    pkl_path = r'../AMASSDataConverter/pkl_data/data_nframe_20_slidewindow_False.pkl'
    sampler = Interpolate(pkl_path=pkl_path,
                     network_path=r'\\winfs-inf\HPS\Mo2Cap2Plus1\work\BodyPoseOptimizer\networks\logs/2_full_dataset_latent_2048_len_20_slide_window_step_1_kl_0.5/checkpoints/14.pth.tar',
                     out_path='out/interpolate2', seq_length=20, latent_dim=2048, windows_size=1)
    for start, end in [(1770, 1300)]:
        print('process: {}'.format((start, end)))
        sampler.out_path = './interpolate_{}_{}'.format(start, end)
        sampler.test(int(start), int(end), visualization=False, save=True)
