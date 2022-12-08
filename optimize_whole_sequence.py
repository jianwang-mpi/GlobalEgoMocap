from optimizer import main
import numpy as np
from natsort import natsorted

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Data directory number')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--camera', required=False, type=str,
                        default='utils/fisheye/fisheye.calibration.json')

    parser.add_argument('--vae', required=False, type=float, default=0.00)
    parser.add_argument('--gmm', required=False, type=float, default=0.00)
    parser.add_argument('--smooth', required=False, type=float, default=0.001)
    parser.add_argument('--bone_length', required=False, type=float, default=0.01)
    parser.add_argument('--weight_3d', required=False, type=float, default=0.01)
    parser.add_argument('--reproj_weight', required=False, type=float, default=0.01)
    parser.add_argument('--save', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--final_smooth', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--merge', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    data_dir = args.data_path

    estimated_pose = []
    optimized_pose = []
    gt_pose = []

    original_global_mpjpe = []
    mid_global_mpjpe = []
    optimized_global_mpjpe = []
    original_camera_pos_error = []
    optimized_camera_pos_error = []
    original_aligned_camera_pos_error = []
    optimized_aligned_camera_pos_error = []
    original_aligned_global_mpjpe = []
    aligned_mid_seq_mpjpe = []
    optimized_aligned_global_mpjpe = []
    aligned_original_mpjpe = []
    aligned_mid_optimized_mpjpe = []
    aligned_optimized_mpjpe = []
    bone_length_aligned_original_mpjpe = []
    bone_length_aligned_mid_optimized_mpjpe = []
    bone_length_aligned_optimized_mpjpe = []
    joints_error = []
    for data_path_name in natsorted(os.listdir(data_dir)):
        data_path = os.path.join(data_dir, data_path_name)
        print("running data: {}".format(data_path))
        if not os.path.isdir(data_path):
            continue
        bone_length = args.bone_length
        reproj_weight = args.reproj_weight
        res, estimated_pose_seq, mid_local_pose_seq, optimized_pose_seq, gt_pose_seq = main(data_path,
                                                                        camera_model_path=args.camera,
                                                                        vae_weight=args.vae, gmm_weight=args.gmm,
                                                                        smoothness_weight=args.smooth,
                                                                        visualization=False, save=args.save,
                                                                        bone_length_weight=bone_length,
                                                                        weight_3d=args.weight_3d,
                                                                        reproj_weight=reproj_weight, merge=args.merge,
                                                                        final_smooth=args.final_smooth)

        estimated_pose.extend(estimated_pose_seq)
        optimized_pose.extend(optimized_pose_seq)
        gt_pose.extend(gt_pose_seq)

        original_global_mpjpe.append(res['original_global_mpjpe'])
        mid_global_mpjpe.append(res['mid_global_mpjpe'])
        optimized_global_mpjpe.append(res['optimized_global_mpjpe'])
        original_camera_pos_error.append(res['original_camera_pos_error'])
        optimized_camera_pos_error.append(res['optimized_camera_pos_error'])
        original_aligned_camera_pos_error.append(res['original_aligned_camera_pos_error'])
        optimized_aligned_camera_pos_error.append(res['optimized_aligned_camera_pos_error'])
        original_aligned_global_mpjpe.append(res['original_aligned_global_mpjpe'])
        aligned_mid_seq_mpjpe.append(res['aligned_mid_seq_mpjpe'])
        optimized_aligned_global_mpjpe.append(res['optimized_aligned_global_mpjpe'])
        aligned_original_mpjpe.append(res['aligned_original_mpjpe'])
        aligned_mid_optimized_mpjpe.append(res['aligned_mid_optimized_mpjpe'])
        aligned_optimized_mpjpe.append(res['aligned_optimized_mpjpe'])
        bone_length_aligned_original_mpjpe.append(res['bone_length_aligned_original_mpjpe'])
        bone_length_aligned_mid_optimized_mpjpe.append(res['bone_length_aligned_mid_optimized_mpjpe'])
        bone_length_aligned_optimized_mpjpe.append(res['bone_length_aligned_optimized_mpjpe'])
        joints_error.append(res['joints_error'])

        if res['bone_length_aligned_optimized_mpjpe'] > res['bone_length_aligned_mid_optimized_mpjpe']:
            print(res)

    print('Average original global pose mpjpe: {}'.format(np.average(original_global_mpjpe)))
    print('Average mid global pose mpjpe: {}'.format(np.average(mid_global_mpjpe)))
    print('Average optimized global pose mpjpe: {}'.format(np.average(optimized_global_mpjpe)))
    print('-----------------------------------------')
    print('Average original cam pose error: {}'.format(np.average(original_camera_pos_error)))
    print('Average optimized cam pose error: {}'.format(np.average(optimized_camera_pos_error)))
    print('-----------------------------------------')
    print('Average original aligned cam pose error: {}'.format(np.average(original_aligned_camera_pos_error)))
    print('Average optimized aligned cam pose error: {}'.format(np.average(optimized_aligned_camera_pos_error)))
    print('-----------------------------------------')
    print('Average original_aligned_global_mpjpe: {}'.format(np.average(original_aligned_global_mpjpe)))
    print('Average aligned_mid_seq_mpjpe: {}'.format(np.average(aligned_mid_seq_mpjpe)))
    print('Average optimized_aligned_global_mpjpe: {}'.format(np.average(optimized_aligned_global_mpjpe)))
    print('-----------------------------------------')
    print('Average aligned original global pose mpjpe: {}'.format(np.average(aligned_original_mpjpe)))
    print('Average aligned mid local pose mpjpe: {}'.format(np.average(aligned_mid_optimized_mpjpe)))
    print('Average aligned optimized global pose mpjpe: {}'.format(np.average(aligned_optimized_mpjpe)))
    print('-----------------------------------------')
    print('Average bone length aligned original global pose mpjpe: {}'.format(
        np.average(bone_length_aligned_original_mpjpe)))
    print('Average bone length aligned mid local pose mpjpe: {}'.format(
        np.average(bone_length_aligned_mid_optimized_mpjpe)))
    print('Average bone length aligned optimized global pose mpjpe: {}'.format(
        np.average(bone_length_aligned_optimized_mpjpe)))
    print('-----------------------------------------')
    print('joints error is: {}'.format(np.mean(joints_error, axis=0)))

    print('-------------------------------------------------------------')
