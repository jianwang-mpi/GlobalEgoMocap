import open3d
import sys
sys.path.append('..')
import torch
from models.SeqConvVAE import ConvVAE
import pickle
from utils.skeleton import Skeleton
import os
from tqdm import tqdm

class Sample:
    def __init__(self, network_path, out_path, seq_length=20, latent_dim=2048):
        self.seq_length = seq_length
        self.out_path = out_path
        self.network = ConvVAE(in_channels=45, out_channels=45, latent_dim=latent_dim, seq_len=self.seq_length)
        state_dict = torch.load(network_path)['state_dict']
        self.network.load_state_dict(state_dict)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.network = self.network.to(self.device)
        
        self.skeleton_model = Skeleton(None)
        
    def save_sample(self, sample, out_path):
        skeleton_list = self.skeleton_model.joint_list_2_mesh_list(sample)
        for i, skeleton in enumerate(skeleton_list):
            open3d.io.write_triangle_mesh(os.path.join(out_path, '{}.ply'.format(i)), skeleton)
            
    def show_example(self, sample):
        skeleton_list = self.skeleton_model.joint_list_2_mesh_list(sample)
        open3d.visualization.draw_geometries(skeleton_list)
        
    def sample(self, sample_num=10, visualization=False):
        sampled_result = self.network.sample(sample_num, current_device=self.device)
        sampled_result = sampled_result.cpu().detach_().numpy()
        sampled_result = sampled_result.reshape((sample_num, self.seq_length, 15, 3))
        for i, sample in tqdm(enumerate(sampled_result)):
            print('sample: {}'.format(i))
            if visualization is True:
                self.show_example(sample)
            else:
                out_path = os.path.join(self.out_path, 'sample_{}'.format(i))
                if not os.path.isdir(out_path):
                    os.mkdir(out_path)
                self.show_example(sample)
                self.save_sample(sample, out_path)
        
if __name__ == '__main__':
    
    # sampler = Sample(network_path='logs/test/checkpoints/37.pth.tar',
    #                  out_path='out/sample')
    sampler = Sample(network_path='logs/only_local_full_dataset_latent_2048_len_10_kl_0.5_2/checkpoints/19.pth.tar',
                     out_path='out/sample/slide_window_latent_2048_len_5_windows_size_5_kl_0.5', seq_length=10,
                     latent_dim=2048)
    sampler.sample(sample_num=12, visualization=True)
