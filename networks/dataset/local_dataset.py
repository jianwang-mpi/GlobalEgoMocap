import sys
sys.path.append('../..')
import open3d

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pickle
import os
from utils.utils import get_relative_global_pose_with_camera_matrix
import numpy as np


class AMASSDataset(Dataset):
    def __init__(self, data_path, frame_num, windows_size=1, is_train=True, fps=25, slide_window=True,
                 balance_distrib=False, with_mo2cap2_data=False):
        """

        :param data_path: directory with pkls containing data
        :param frame_num: length of sequence
        :param windows_size: one frame from n frame
        :param is_train:
        :param fps: the fps of test data
        """
        self.windows_size = windows_size
        self.is_train = is_train
        self.slide_window = slide_window
        self.frame_num = frame_num
        self.balance_distrib = balance_distrib
        self.with_mo2cap2_data = with_mo2cap2_data
        self.data_list = self.load_pkls(data_path, is_train, self.with_mo2cap2_data, self.balance_distrib)
        self.local_pose_seq_list = self.get_local_pose_list(self.data_list, frame_num=self.frame_num,
                                                                         windows_size=self.windows_size,
                                                                         fps=fps)
        # with open(output_path, 'wb') as f:
        #     pickle.dump(self.relative_pose_seq_list, f, protocol=4)
    
    def __getitem__(self, i):
        data_i = self.local_pose_seq_list[i]
        data_i = data_i.reshape((-1, 45))
        data_i = torch.from_numpy(data_i).float()
        return data_i
    
    def __len__(self):
        return len(self.local_pose_seq_list)
    
    def load_pkls(self, path, is_train, with_mo2cap2_data=False, balance_distrib=False):
        data_list = []
        raw_pkl_path_list = os.listdir(path)
        if with_mo2cap2_data is True:
            print('use only mo2cap2 motion!')
            seq_names = np.load('/HPS/ScanNet/static00/SURREAL/smpl_data/seq_names.npy').tolist()
            pkl_path_list = []
            for seq_name in seq_names:
                for pkl_path in raw_pkl_path_list:
                    if seq_name in pkl_path:
                        pkl_path_list.append(pkl_path)
        else:
            pkl_path_list = raw_pkl_path_list
        print(len(pkl_path_list))
        # pkl_path_list = [pkl_path for pkl_path in pkl_path_list if 'CMU' in pkl_path]
        if is_train:
            pkl_path_list = pkl_path_list[:-10]
        else:
            pkl_path_list = pkl_path_list[-10:]
        
        if balance_distrib is True:
            print('use balanced distribution')
            walking_pkl_path_list = [pkl_path for pkl_path in pkl_path_list if 'walk' in pkl_path.lower()]
            non_walking_pkl_path_list = [pkl_path for pkl_path in pkl_path_list if 'walk' not in pkl_path.lower()]
            np.random.shuffle(walking_pkl_path_list)
            walking_num = int(1 / 20 * len(non_walking_pkl_path_list))
            pkl_path_list = non_walking_pkl_path_list + walking_pkl_path_list[:walking_num]
        for file_name in tqdm(pkl_path_list):
            file_path = os.path.join(path, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                data_list.append(data)
        return data_list
    
    def get_local_pose_list(self, data_list, frame_num=10, windows_size=1, fps=30):
        total_frame_number = frame_num * windows_size
        
        local_pose_seq_list = []
        for seq_data in tqdm(data_list):
            seq_len = len(seq_data['local_pose_list'])
            data_frame_rate = int(seq_data['frame_rate'])
            frame_rate_timer = round(data_frame_rate / fps)
            if self.slide_window is True:
                interval = 1
            else:
                interval = total_frame_number * frame_rate_timer
            for i in range(0, seq_len - total_frame_number * frame_rate_timer, interval):
                local_pose_seq_list.append(
                    seq_data['local_pose_list'][i: i + total_frame_number * frame_rate_timer: frame_rate_timer])
        
        return np.asarray(local_pose_seq_list)


class Mo2Cap2Dataset(Dataset):
    def __init__(self, pkl_path, frame_num=5, slide_window=False):
        self.frame_num = frame_num
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            estimated_local_skeleton = data['estimated_local_skeleton']
            gt_skeleton = data['gt_global_skeleton']
            cv2world_mat_list = data['camera_pose_list']
        
        self.estimated_local_pose_list = []
        self.gt_pose_list = []
        self.cam_list = []
        # split to 5-frame sequence
        for i in tqdm(range(0, len(estimated_local_skeleton) - frame_num, frame_num)):
            estimated_seq = estimated_local_skeleton[i: i + self.frame_num]
            cam_seq = cv2world_mat_list[i: i + self.frame_num]
            gt_seq = gt_skeleton[i: i + self.frame_num]
            self.estimated_local_pose_list.append(np.asarray(estimated_seq))
            self.gt_pose_list.append(np.asarray(gt_seq))
            self.cam_list.append(np.asarray(cam_seq))
        
        self.slide_window = False
        
        self.relative_global_pose_list = self.get_relative_global_pose_list(self.estimated_local_pose_list,
                                                                            self.cam_list)
        
        print(len(self.relative_global_pose_list))
    
    def get_relative_global_pose_list(self, estimated_local_pose_list, cam_list):
        relative_global_pose_seq_list = []
        
        for local_pose_seq, camera_seq in tqdm(zip(estimated_local_pose_list, cam_list)):
            relative_global_pose_seq = get_relative_global_pose_with_camera_matrix(local_pose_seq, camera_seq)
            
            # select {frame_num} pose from relative global pose list
            final_relative_global_pose_seq = []
            for i in range(0, len(relative_global_pose_seq)):
                final_relative_global_pose_seq.append(relative_global_pose_seq[i])
            
            relative_global_pose_seq_list.append(final_relative_global_pose_seq)
        
        return np.asarray(relative_global_pose_seq_list)
    
    def __getitem__(self, index):
        out_pose = self.relative_global_pose_list[index].reshape([-1, 45])
        out_pose = torch.from_numpy(out_pose).float()
        
        out_cam = self.cam_list[index]
        out_cam = torch.from_numpy(out_cam).float()
        
        out_gt = self.gt_pose_list[index]
        out_gt = torch.from_numpy(out_gt).float()
        return out_pose, out_cam, out_gt
    
    def __len__(self):
        return len(self.relative_global_pose_list)


if __name__ == '__main__':

    dataset = AMASSDataset(data_path='/home/jianwang/ScanNet/static00/EgocentricAMASS',
                           frame_num=10, windows_size=1,
                           is_train=True)
    print(len(dataset))
    
    # dataset = Mo2Cap2Dataset(pkl_path='../test_data_weipeng_studio/data_start_887_end_987/test_data.pkl')
    # from utils.skeleton import Skeleton
    #
    # skeleton_model = Skeleton(calibration_path=None)
    # d = dataset.relative_global_pose_list[0]
    # skeleton_list = skeleton_model.joint_list_2_mesh_list(d)
    # open3d.visualization.draw_geometries(skeleton_list)
