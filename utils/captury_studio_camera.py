import numpy as np


class CapturyCamera:
    def __init__(self, camera_path, camera_number):
        camera_data = self.load_camera_model(camera_path, camera_number)
        self.intrinsic = camera_data['intrinsic']
        self.extrinsic = camera_data['extrinsic']
        self.distortion = camera_data['distortion']

    def get_camera_model(self):
        return self.intrinsic, self.extrinsic, self.distortion

    def load_camera_model(self, camera_path, camera_number):
        with open(camera_path) as f:
            lines = f.readlines()
        start_line_number = -1
        for i, line in enumerate(lines):
            if 'camera	{}'.format(camera_number) in line:
                start_line_number = i
                break
        if start_line_number == -1:
            print('camera not found')
        camera_lines = lines[start_line_number: start_line_number + 27]
        distortion_line = camera_lines[11]
        extrinsic_lines = camera_lines[73 - 56: 76 - 56]
        intrinsic_lines = camera_lines[77 - 56: 80 - 56]
        distortions = distortion_line.split()[1:]
        distortions = np.asarray(distortions).astype(np.float)
        # print(distortions)
        extrinsic_lines = [line.split()[1:] for line in extrinsic_lines]
        intrinsic_lines = [line.split()[1:] for line in intrinsic_lines]

        extrinsic = np.asarray(extrinsic_lines).astype(np.float)
        intrinsic = np.asarray(intrinsic_lines).astype(np.float)

        return {'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'distortion': distortions}
