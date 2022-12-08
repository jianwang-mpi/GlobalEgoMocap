import open3d
import sys
sys.path.append('..')
import torch
from SeqConvVAE import ConvVAE
import pickle
from utils.skeleton import Skeleton
import os
from tqdm import tqdm
from dataset import AMASSDataset
from dataset import Mo2Cap2Dataset
from torch.utils.data import DataLoader
import numpy as np

class Sample:
    def __init__(self, network_path, out_path, seq_length=20, latent_dim=128):
        self.seq_length = seq_length
        self.out_path = out_path
        self.network = ConvVAE(in_channels=45, out_channels=45, latent_dim=latent_dim, seq_len=self.seq_length)
        state_dict = torch.load(network_path)['state_dict']
        self.network.load_state_dict(state_dict)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.network = self.network.to(self.device)
        
        self.skeleton_model = Skeleton(None)

        # self.test_dataset = AMASSDataset(data_path='/home/wangjian/Develop/BodyPoseOptimization/AMASSDataConverter/pkls',
        #                        frame_num=self.seq_length,
        #                        is_train=False)

        self.test_dataset = Mo2Cap2Dataset(
            pkl_path='../test_data_weipeng_studio/data_start_1187_end_1287/test_data.pkl')

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=4, shuffle=False,
                                          drop_last=False, num_workers=2)
        
    def save_sample(self, sample, out_path):
        skeleton_list = self.skeleton_model.joint_list_2_mesh_list(sample)
        for i, skeleton in enumerate(skeleton_list):
            open3d.io.write_triangle_mesh(os.path.join(out_path, '{}.ply'.format(i)), skeleton)
            
    def show_example(self, sample):
        skeleton_list = self.skeleton_model.joint_list_2_mesh_list(sample)
        open3d.visualization.draw_geometries(skeleton_list)
        
    def test(self, visualization=False):
        print('---------------------Start Eval-----------------------')
        self.network.eval()
        with torch.no_grad():
            for i, relative_global_pose in tqdm(enumerate(self.test_dataloader)):
                relative_global_pose = relative_global_pose.to(self.device)
                relative_global_pose = relative_global_pose
                mu, std, z = self.network.get_latent_space(relative_global_pose)
                # print(mu)
                # print(std)
                #
                print('mu error is: {}'.format(torch.sum(torch.square(mu))))
                print('std error is: {}'.format(torch.sum(torch.square(std - 1))))
                print('---------------------------------------')
                

                pose_preds, pose_inputs, _, _ = self.network(relative_global_pose)
                mu, std, z = self.network.get_latent_space(pose_preds)
                print('vae refined mu error is: {}'.format(torch.sum(torch.square(mu))))
                print('vae refined std error is: {}'.format(torch.sum(torch.square(std - 1))))
                print('***********************************************')
                

        
if __name__ == '__main__':
    
    # sampler = Sample(network_path='logs/test/checkpoints/37.pth.tar',
    #                  out_path='out/sample')
    sampler = Sample(network_path='logs/real_full_dataset_latent_2048_len_5_slide_window_step_1_kl_0.5/checkpoints/19.pth.tar',
                     out_path='out/test', seq_length=5, latent_dim=2048)
    sampler.test(visualization=True)
