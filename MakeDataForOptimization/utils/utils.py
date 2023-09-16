import datetime
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation

def transformation_matrix_to_translation_and_rotation(mat):
    rotation_matrix = mat[:3, :3]
    translation = mat[:3, 3]

    # rotation matrix to rotation euler
    rotation_euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
    return rotation_euler, translation


def get_relative_transform(location1, rotation1, location2, rotation2):
    # input: location and rotation in blender coordinate
    # out: rotation, translation and transform matrix in OpenCV coordinate
    T_world2cv1, R_world2cv1, mat_world2cv1 = get_cv_rt_from_blender(location1, rotation1)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location2, rotation2)

    mat_cv1_2world = np.linalg.inv(mat_world2cv1)

    mat_cv1_to_cv2 = mat_cv1_2world.dot(mat_world2cv2)
    # mat cv1 to cv2 is the coordinate transformation, we need to change it to the object transformation
    mat_cv2_to_cv1 = np.linalg.inv(mat_cv1_to_cv2)

    rotation, translation = transformation_matrix_to_translation_and_rotation(mat_cv2_to_cv1)
    return rotation, translation, mat_cv2_to_cv1


def get_transform_relative_to_base_cv(base_location, base_rotation, location, rotation):
    # input: location and rotation in blender coordinate
    # out: rotation, translation and transform matrix in OpenCV coordinate
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_cv(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_cv(location, rotation)

    # mat_cv2world2 = np.linalg.inv(mat_world2cv2)
    # location_cv_homo = mat_cv2world2[:, 3]

    location_cv_homo = np.concatenate([location, np.ones(1)])

    R_cv2_2_base = R_world2cv2.T.dot(R_world2cv_base)
    new_rotation_euler = Rotation.from_matrix(R_cv2_2_base).as_euler(seq='xyz')

    new_location_homo = mat_world2cv_base.dot(location_cv_homo)
    new_location = new_location_homo[:3]

    return new_location, new_rotation_euler

def get_transform_relative_to_base_blender(base_location, base_rotation, location, rotation):
    T_world2cv_base, R_world2cv_base, mat_world2cv_base = get_cv_rt_from_blender(base_location, base_rotation)
    T_world2cv2, R_world2cv2, mat_world2cv2 = get_cv_rt_from_blender(location, rotation)

    location_cv_homo = np.concatenate([location, np.ones(1)])

    R_cv2_2_base = R_world2cv2.T.dot(R_world2cv_base)
    new_rotation_euler = Rotation.from_matrix(R_cv2_2_base).as_euler(seq='xyz')

    new_location_homo = mat_world2cv_base.dot(location_cv_homo)
    new_location = new_location_homo[:3]

    return new_location, new_rotation_euler

# code modified from zaw lin
def get_cv_rt_from_blender(location, rotation):
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]])

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2bcam = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam.dot(location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv.dot(R_world2bcam)
    T_world2cv = R_bcam2cv.dot(T_world2bcam)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat

# code modified from zaw lin
def get_cv_rt_from_cv(location, rotation):

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints

    R_world2cv = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix().T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2cv = -1 * R_world2cv.dot(location)

    #put into 3x4 matrix
    mat = np.array([
        np.concatenate([R_world2cv[0], [T_world2cv[0]]]),
        np.concatenate([R_world2cv[1], [T_world2cv[1]]]),
        np.concatenate([R_world2cv[2], [T_world2cv[2]]]),
        [0, 0, 0, 1]
    ])
    return T_world2cv, R_world2cv, mat