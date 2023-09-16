import open3d
from utils.skeleton import Skeleton
from npybvh.bvh import Bvh
import numpy as np
from tqdm import tqdm
import pickle

skeleton_model = Skeleton(None)
anim = Bvh()
egocentric_joints = [6, 15, 16, 17, 10, 11, 12, 23, 24, 25, 26, 19, 20, 21, 22]


def parse_file(bvh_file_path, output_file_path, start_frame, input_frame_rate, output_frame_rate):
    anim.parse_file(bvh_file_path)
    gt_pose_seq = []
    print(anim.frames)
    print(anim.joint_names())
    step = round(input_frame_rate / output_frame_rate)
    for frame in tqdm(range(start_frame, anim.frames, step)):
        positions, rotations = anim.frame_pose(frame)

        positions = positions[egocentric_joints]
        positions = positions / 1000
        gt_pose_seq.append(positions)

        skeleton = skeleton_model.joints_2_mesh(positions)

        open3d.visualization.draw_geometries([skeleton])
    gt_pose_seq = np.asarray(gt_pose_seq)
    # skeleton_list = skeleton_model.joint_list_2_mesh_list(gt_pose_seq)
    # open3d.visualization.draw_geometries(skeleton_list)
    with open(output_file_path, 'wb') as f:
        pickle.dump(gt_pose_seq, f)


if __name__ == '__main__':
    parse_file(r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\CapturyData\GoProResults\captury\unknown.bvh',
               'data/wild/wild.pkl', start_frame=0, input_frame_rate=50, output_frame_rate=25)