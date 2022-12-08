import numpy as np

def mpjpe_on_different_joints(pose_preds, pose_gt):
    distance = np.linalg.norm(pose_gt - pose_preds, axis=-1)
    distance_on_each_joints = np.mean(distance, axis=tuple(range(distance.ndim - 1)))
    return distance_on_each_joints