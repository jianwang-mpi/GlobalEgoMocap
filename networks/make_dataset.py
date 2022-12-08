import open3d
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import sys
sys.path.append('..')
from tqdm import tqdm
import pickle
import os
from utils.utils import get_relative_global_pose, get_relative_global_pose_with_camera_matrix, trans_qrot_to_matrix
import numpy as np
import h5py


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    
    def __init__(self, datapath, dataset_shape_dict: dict, dtype=np.float32):
        self.datapath = datapath
        self.dataset_shape_dict = dataset_shape_dict
        
        with h5py.File(self.datapath, mode='w') as h5f:
            for dataset_name in self.dataset_shape_dict.keys():
                h5f.create_dataset(
                    dataset_name,
                    shape=(0,) + self.dataset_shape_dict[dataset_name],
                    maxshape=(None,) + self.dataset_shape_dict[dataset_name],
                    dtype=dtype)
    
    def append(self, values_batch_dict: dict):
        # print('append data!')
        with h5py.File(self.datapath, mode='a') as h5f:
            for dataset_name, values_batch in values_batch_dict.items():
                batch_size = values_batch.shape[0]
                dset = h5f[dataset_name]
                dset_len = dset.shape[0]
                shape = dset.shape[1:]
                dset.resize((dset_len + batch_size,) + shape)
                for i in range(dset_len, dset_len + batch_size):
                    dset[i] = [values_batch[i - dset_len]]
            h5f.flush()


def main(source_dir, output_path, frame_num, fps, slide_window):
    pkl_path_list = os.listdir(source_dir)
    dataset_shape_dict = {"relative_global_pose": (frame_num, 15, 3),
                          "local_pose": (frame_num, 15, 3),
                          "camera_matrix": (frame_num, 4, 4)}
    hdf5_store = HDF5Store(output_path, dataset_shape_dict)
    for file_name in tqdm(pkl_path_list):
        file_path = os.path.join(source_dir, file_name)
        relative_global_pose_seq_list, local_pose_seq_list, camera_matrix_seq_list = \
            get_relative_global_pose_list(file_path, frame_num, fps, slide_window)
        values_batch_dict = {"relative_global_pose": relative_global_pose_seq_list,
                          "local_pose": local_pose_seq_list,
                          "camera_matrix": camera_matrix_seq_list}
        hdf5_store.append(values_batch_dict)
                
        
def interpolate(original_sequence, interpolate_time=5):
    result_list = []
    for i in range(len(original_sequence) - 1):
        local_pose_i = original_sequence[i]
        local_pose_i_1 = original_sequence[i + 1]
        diff = local_pose_i_1 - local_pose_i
        res = np.empty(shape=(interpolate_time, ) + local_pose_i.shape)
        for j in range(interpolate_time):
            res[j] = local_pose_i + diff * j / interpolate_time
        result_list.append(res)
    return np.concatenate(result_list, axis=0)

def get_relative_global_pose_list(data_path, frame_num=5, fps=25, slide_window=True):
    total_frame_number = frame_num
    relative_global_pose_seq_list = []
    
    local_pose_seq_list = []
    camera_seq_list = []
    camera_matrix_seq_list = []
    with open(data_path, 'rb') as f:
        seq_data = pickle.load(f)
    seq_len = len(seq_data['local_pose_list'])
    data_frame_rate = int(seq_data['frame_rate'])
    frame_rate_timer = round(data_frame_rate / fps)
    if slide_window is True:
        interval = 1
    else:
        interval = total_frame_number * frame_rate_timer
    for i in range(0, seq_len - total_frame_number * frame_rate_timer, interval):
        local_pose_seq_list.append(
            seq_data['local_pose_list'][i: i + total_frame_number * frame_rate_timer: frame_rate_timer])
        camera_seq_list.append(
            seq_data['cam_list'][i: i + total_frame_number * frame_rate_timer: frame_rate_timer])
    for local_pose_seq, camera_seq in zip(local_pose_seq_list, camera_seq_list):
        relative_global_pose_seq = get_relative_global_pose(local_pose_seq, camera_seq)
        camera_matrix_seq = []
        for camera in camera_seq:
            camera_matrix = trans_qrot_to_matrix(camera['loc'], camera['rot'])
            camera_matrix_seq.append(camera_matrix)
        camera_matrix_seq_list.append(camera_matrix_seq)
        # select {frame_num} pose from relative global pose list
        final_relative_global_pose_seq = []
        for i in range(0, len(relative_global_pose_seq)):
            final_relative_global_pose_seq.append(relative_global_pose_seq[i])
        
        relative_global_pose_seq_list.append(final_relative_global_pose_seq)
    
    return np.asarray(relative_global_pose_seq_list), np.asarray(local_pose_seq_list), np.asarray(camera_matrix_seq_list)


if __name__ == '__main__':
    source_dir = '/home/jianwang/ScanNet/static00/EgocentricAMASS'
    
    frame_num, fps, slide_window = 10, 25, True
    target_file_path = '/home/jianwang/ScanNet/static00/EgocentricAMASSPytorch/local_relative_global_pose_{}_{}.h5'.format(frame_num, fps)
    main(source_dir, target_file_path, frame_num, fps, slide_window)