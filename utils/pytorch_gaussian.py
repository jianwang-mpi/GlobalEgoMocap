import torch
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch.nn.functional as F



def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

if __name__ == '__main__':
    pose_random = np.random.randn()
    smoothed_skeleton_seq_np = gaussian_filter1d(pose_random, sigma=1)
    print(smoothed_skeleton_seq_np)
    
    
    pose_random_torch = torch.from_numpy(pose_random).unsqueeze_(0).unsqueeze_(0).float()
    kernel = gaussian(window_size=5, sigma=1)
    kernel = kernel.unsqueeze_(0).unsqueeze_(0).float()
    
    smoothed_skeleton_seq_torch = F.conv1d(pose_random_torch, kernel)
    print(smoothed_skeleton_seq_torch)