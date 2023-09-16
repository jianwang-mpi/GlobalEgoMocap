import open3d
from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
from utils.captury_studio_camera import CapturyCamera
from utils.skeleton import Skeleton
import numpy as np

import pickle
import torch
import cv2
from copy import deepcopy

from utils.utils import draw_joints

def process_2D_pose(raw_pose):
    if raw_pose is None:
        pose_egopose_model = np.zeros(shape=(15, 3))
    else:
        pose = []
        for i in range(0, len(raw_pose), 3):
            x = raw_pose[i]
            y = raw_pose[i + 1]
            confidence = raw_pose[i + 2]
            pose.append(np.asarray((x, y, confidence)))
        neck = pose[1] + (pose[0] - pose[1]) * 0.25
        # neck = pose[1]
        pose_egopose_model = [neck, pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], pose[9], pose[10], pose[11],
                              pose[22], pose[12], pose[13], pose[14], pose[19]]
    return np.asarray(pose_egopose_model)


class FisheyeEpipolarGeometry:
    def __init__(self):
        self.skeleton_model = Skeleton(None)

    def get_extrinsic_matrix(self, R, t):
        # output: matrix 3 * 4
        extrinsic_matrix = np.empty(shape=(3, 4))
        extrinsic_matrix[:, :3] = R
        extrinsic_matrix[:, 3] = t.reshape(3)
        return extrinsic_matrix

    def get_projection_matrix(self, K, R, t):
        extrinsic_matrix = self.get_extrinsic_matrix(R, t)
        P = K @ extrinsic_matrix
        return P

    def depth(self, point3D, R, t):
        R = np.asarray(R)
        t = np.asarray(t)
        res = (R @ point3D.T)[2] + t[2]
        return res.reshape(point3D.shape[0])

    def camera_pose_from_essential(self, essential_matrix):
        u, s, vt = np.linalg.svd(essential_matrix)
        if np.linalg.det(u) < 0:
            u[:, 2] *= -1.0
        if np.linalg.det(vt) < 0:
            vt[2] *= -1.0

        # Find R and T from Hartley & Zisserman
        W = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

        U_W_Vt = u @ W @ vt
        U_Wt_Vt = u @ W.T @ vt

        result = [
            (U_W_Vt, u[:, 2]),
            (U_W_Vt, -u[:, 2]),
            (U_Wt_Vt, u[:, 2]),
            (U_Wt_Vt, -u[:, 2])
        ]
        return result

    def select_camera_pose(self, possible_camera_poses, intrinsic_K_1, intrinsic_K_2, points2D_1, points2D_2):

        R1, t1 = np.eye(3), np.zeros(3)
        P1 = self.get_projection_matrix(intrinsic_K_1, R1, t1)
        for R, t in possible_camera_poses:
            P2 = self.get_projection_matrix(intrinsic_K_2, R, t)

            points3D_homo_list = cv2.triangulatePoints(P1, P2, points2D_1.T, points2D_2.T)
            points3D_homo_list = points3D_homo_list.T
            points3D = points3D_homo_list[:, :3] / points3D_homo_list[:, 3:]
            d1 = self.depth(points3D, R1, t1)
            d2 = self.depth(points3D, R, t)

            if (d1[-6:] > 0).all() and (d2[-6:] > 0).all():
                return R, t, points3D

        return None, None, None

    def get_camera_pose(self, points_1, points_2, camera_matrix_1, camera_matrix_2):
        candidate_points_1 = [points_1[i][:2] for i in range(len(points_1)) if points_2[i][2] > 0.6]
        candidate_points_1 = np.asarray(candidate_points_1)
        candidate_points_2 = [points_2[i][:2] for i in range(len(points_2)) if points_2[i][2] > 0.6]
        candidate_points_2 = np.asarray(candidate_points_2)


        F, mask = cv2.findFundamentalMat(candidate_points_1, candidate_points_2, cv2.FM_RANSAC)
        K1 = np.asarray(camera_matrix_1)
        K2 = np.asarray(camera_matrix_2)
        E = K2.T @ np.mat(F) @ K1

        possible_camera_poses = self.camera_pose_from_essential(E)

        final_camera_pose_R, final_camera_pose_t, point3D = self.select_camera_pose(possible_camera_poses, camera_matrix_1, camera_matrix_2,
                                                    candidate_points_1, candidate_points_2)
        return final_camera_pose_R, final_camera_pose_t, point3D

    def get_camera_pose_fisheye_pinhole(self, points_fisheye, points_pinhole, fisheye_camera: FishEyeCameraCalibrated,
                                        pinhole_camera_matrix):
        undistorted_points = fisheye_camera.undistort(points_fisheye)
        fisheye_intrinsic_matrix = fisheye_camera.intrinsic[:3, :3]

        # pinhole_camera_matrix[:2] *= 808

        relative_camera_r, relative_camera_t, point3D = self.get_camera_pose(undistorted_points, points_pinhole,
                                                                    fisheye_intrinsic_matrix, pinhole_camera_matrix)

        return relative_camera_r, relative_camera_t


    def get_camera_pose_fisheye_pinhole_test(self, points_fisheye, points_pinhole, fisheye_camera: FishEyeCameraCalibrated,
                                             pinhole_camera_matrix, local_point_3d):
        undistorted_points = fisheye_camera.undistort(points_fisheye)
        fisheye_intrinsic_matrix = fisheye_camera.intrinsic[:3, :3]

        pinhole_camera_matrix[:2] *= 808



        relative_camera_r, relative_camera_t, point3D = self.get_camera_pose(undistorted_points, points_pinhole,
                                                                    fisheye_intrinsic_matrix, pinhole_camera_matrix)

        # point3D = point3D * 10
        relative_camera_t = relative_camera_t
        point3D = local_point_3d * (1/3)

        # skeleton_model = Skeleton(calibration_path=None)
        # skeleton_mesh = skeleton_model.joints_2_mesh(local_point_3d)
        # coor = open3d.geometry.TriangleMesh.create_coordinate_frame()
        # open3d.visualization.draw_geometries([skeleton_mesh, coor])

        # draw projected 3D points under egocentric camera
        # projected_points2D = fisheye_camera.world2camera(point3D)
        # # #
        # # # img = np.zeros((1024, 1280, 3))
        # img = cv2.imread(r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\EgocentricData\REC08102020\jian3\imgs\img-10082020170746-557.jpg')
        # img = draw_joints(projected_points2D, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # draw projected 3d points under external camera

        external_camera_projection_matrix = self.get_projection_matrix(pinhole_camera_matrix,
                                                                       relative_camera_r, relative_camera_t)
        point3D_homo = np.ones((point3D.shape[0], 4))
        point3D_homo[:, :3] = point3D
        projected_2D_points = (external_camera_projection_matrix @ point3D_homo.T).T
        projected_2D_points = projected_2D_points[:, :2] / projected_2D_points[:, 2:]

        img = cv2.imread(r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\CapturyData\REC08102020\jian3\stream03\img_000320.jpg')
        img = draw_joints(projected_2D_points, img)
        cv2.imshow('img', img)
        cv2.waitKey(0)

        return relative_camera_r, relative_camera_t



if __name__ == '__main__':

    test_data_path = r'X:\Mo2Cap2Plus\work\MakeWeipengStudioTestData\external_view_data\jian3_external\data_start_557_end_657'

    fisheye_camera = FishEyeCameraCalibrated('fisheye.calibration.json')
    external_camera = CapturyCamera(camera_path=r'X:\Mo2Cap2Plus1\static00\CapturyData\REC08102020\jian3\camera.calib',
                                    camera_number=3)


    with open('{}/test_data.pkl'.format(test_data_path), 'rb') as f:
        data = pickle.load(f)
        estimated_local_skeleton = data['estimated_local_skeleton']
        gt_skeleton = data['gt_global_skeleton']
        cv2world_mat_list = data['camera_pose_list']
        heatmap_list = data['heatmap_list']
        pose_external_2D_list = data['external_2D_prediction']
        external_camera_param = data['external_camera']
        estimated_local_skeleton = np.asarray(estimated_local_skeleton)
        gt_skeleton = np.asarray(gt_skeleton)
        cv2world_mat_list = np.asarray(cv2world_mat_list)
        heatmap_list = np.asarray(heatmap_list)

    for i in range(len(pose_external_2D_list)):
        pose_external_2D_list[i] = process_2D_pose(pose_external_2D_list[i])

    epipolar_geometry = FisheyeEpipolarGeometry()

    fisheye_pose_2d = fisheye_camera.world2camera(deepcopy(estimated_local_skeleton).reshape((-1, 3)))
    fisheye_pose_2d = fisheye_pose_2d.reshape((-1, 15, 2))

    # external_pose_2d = pose_external_2D_list

    pose_external_2D_list = np.asarray(pose_external_2D_list).reshape((100, 15, 3))
    # pose_external_2D_list = pose_external_2D_list[:, :, :2]

    i = 1

    relative_camera_r, relative_camera_t = epipolar_geometry.get_camera_pose_fisheye_pinhole_test(fisheye_pose_2d[i],
                                                                                                  pose_external_2D_list[i],
                                                                                                  fisheye_camera,
                                                                                                  external_camera_param['intrinsic'],
                                                                                                  estimated_local_skeleton[i])

    print(relative_camera_r)
    print(relative_camera_t)



# if __name__ == '__main__':
#     fisheye_epipolar = FisheyeEpipolarGeometry()
#
#     camera_intrinsic_matrix1 = np.asarray([[1, 0, 256],
#                                            [0, 1, 256],
#                                            [0, 0, 1]])
#     camera_intrinsic_matrix2 = np.asarray([[1, 0, 256],
#                                            [0, 1, 256],
#                                            [0, 0, 1]])
#
#     camera_R1 = np.eye(3)
#     camera_t1 = np.asarray([0, 0, 5])
#
#     camera_R2 = np.asarray([[1, 0, 0],
#                             [0, 0, -1],
#                             [0, 1, 0]])
#     camera_t2 = np.asarray([0, 0, 5])
#
#     point3Ds = np.asarray([[0, 0, 0, 1],
#                            [1, 0, 0, 1],
#                            [0, 1, 0, 1],
#                            [0, 0, 1, 1],
#                            [1, 1, 0, 1],
#                            [0, 1, 1, 1],
#                            [1, 0, 1, 1],
#                            [1, 1, 1, 1],
#                            [-1, 0, 1, 1],
#                            [1, 0, -1, 1],
#                            [-1, 0, -1, 1],
#                            [-1, -1, -1, 1],])
#
#
#     coordinate_zero = open3d.geometry.TriangleMesh.create_coordinate_frame()
#     camera_2_open3d: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_coordinate_frame()
#     camera_2_open3d.rotate(camera_R2)
#     camera_2_open3d.translate(camera_t2)
#
#     # open3d.visualization.draw_geometries([coordinate_zero, camera_2_open3d])
#
#     projection_matrix_1 = fisheye_epipolar.get_projection_matrix(camera_intrinsic_matrix1, camera_R1, camera_t1)
#     projection_matrix_2 = fisheye_epipolar.get_projection_matrix(camera_intrinsic_matrix2, camera_R2, camera_t2)
#
#     full_projection_matrix_1 = np.eye(4)
#     full_projection_matrix_1[:3, :] = projection_matrix_1
#
#     full_projection_matrix_2 = np.eye(4)
#     full_projection_matrix_2[:3, :] = projection_matrix_2
#
#     print(full_projection_matrix_2 @ np.linalg.inv(full_projection_matrix_1))
#
#
#     point2D_1 = (projection_matrix_1 @ point3Ds.T).T
#     point2D_1 = point2D_1 / point2D_1[:, 2:]
#     point2D_1 = point2D_1[:, :2]
#     point2D_2 = (projection_matrix_2 @ point3Ds.T).T
#     point2D_2 = point2D_2 / point2D_2[:, 2:]
#     point2D_2 = point2D_2[:, :2]
#
#
#
#     print(point2D_1)
#     print(point2D_2)
#
#     # point2D_1 = np.asarray([[100, 100]])
#     # point2D_2 = np.asarray([[300, 300]])
#
#     final_camera_pose = fisheye_epipolar.get_camera_pose(point2D_1, point2D_2,
#                                                          camera_intrinsic_matrix1, camera_intrinsic_matrix2)
#     print(final_camera_pose)
