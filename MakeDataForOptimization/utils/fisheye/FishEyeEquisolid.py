import json
import numpy as np
import torch


class FishEyeCameraEquisolid:
    def __init__(self, focal_length, sensor_size, img_size, use_gpu=False):
        """
        @param    focal_length: focal length of camera in mm
        @param    sensor_size: sensor size of camera in mm
        @param    img_size: image size of w, h
        @param    use_gpu: whether use the gpu to accelerate the calculation
        """
        self.sensor_size = sensor_size
        self.img_size = np.asarray(img_size)
        self.use_gpu = use_gpu
        # calculate the focal length in pixel
        self.focal_length = focal_length / np.max(sensor_size) * np.max(img_size)

        # calculate the image center
        self.img_center = self.img_size / 2 + 1e-10
        # get max distance from image center
        self.max_distance = self.focal_length * np.sqrt(2)

        if self.use_gpu:
            gpu_ok = torch.cuda.is_available()
            if gpu_ok is False:
                raise Exception("GPU is not available!")

    def camera2world(self, point: np.ndarray, depth: np.ndarray):
        """
        @param point: 2d point in shape: (n, 2)
        @param depth: depth of every points
        @return: 3d position of every point
        """
        # get the distance of point to center
        depth = depth.astype(np.float)
        point_centered = point.astype(np.float) - self.img_center
        x = point_centered[:, 0]
        y = point_centered[:, 1]
        distance_from_center = np.sqrt(np.square(x) + np.square(y))
        distance_from_center[distance_from_center > self.max_distance - 30] = self.max_distance

        theta = 2 * np.arcsin(distance_from_center / (2 * self.focal_length))
        Z = distance_from_center / np.tan(theta)

        # square_sin_theta_div_2 = np.square(distance_from_center / (2 * self.focal_length))
        # tan_theta_div_1 = np.sqrt(1 / (4 * square_sin_theta_div_2 * (1 - square_sin_theta_div_2)) - 1)
        # Z = distance_from_center * tan_theta_div_1
        point_3d = np.array([x, y, Z])
        norm = np.linalg.norm(point_3d, axis=0)
        point_3d = point_3d / norm * depth
        return point_3d.transpose()

    def camera2world_pytorch(self, point: torch.Tensor(), depth: torch.Tensor):
        pass


if __name__ == '__main__':
    camera = FishEyeCameraEquisolid(focal_length=9, sensor_size=32, img_size=(1280, 1024))
    point = np.array([[660, 20], [660, 20], ])
    depth = np.array([10, 10])
    point3d = camera.camera2world(point, depth)
    print(point3d)
