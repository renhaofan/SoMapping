"""
@author steve
@brief Generate TUM format ground truth trajectory from pose folder.
@details tx,ty,tz, qx, qy, qz, qr. Real part of quaternion is located in last one.
"""
import numpy as np
import os
from scipy.spatial.transform import Rotation

scene_path = '/home/steve/dataset/scene0427_00'
pose_folder = os.path.join(scene_path, 'pose')

with open(os.path.join(scene_path, 'scene0427_00.txt'), 'r') as f:
    list = f.read()

list_split = list.split()[2:18]
axisAlignment = np.zeros((4, 4), dtype=np.float64)
frame_transform = np.zeros((4, 4), dtype=np.float64)

frame_transform[0][0] = -1
frame_transform[1][2] = -1
frame_transform[2][1] = -1
frame_transform[3][3] = 1

k = 0
for i in range(4):
    for j in range(4):
        axisAlignment[i][j] = list_split[k]
        k = k + 1
print(axisAlignment)
print(frame_transform)

with open(os.path.join(scene_path, 'groundtruth_opengl_float.txt'), 'x') as f:
    f.write('# ground truth trajectory\n')
    f.write('# scene: ' + scene_path.rstrip('/').split('/')[-1] + '\n')
    f.write('# fake_timestamp tx ty tz qx qy qz qw ' + '\n')
    for _, _, files in os.walk(pose_folder):
        files.sort(key = lambda x: int(x[:-4]))
        print(f'Pose file number: {len(files)}')
        for file in files:
            # extract quaternion from pose
            pose_path = os.path.join(pose_folder, file)
            pose = np.loadtxt(pose_path)
            # transform because opengl
            pose = np.matmul(axisAlignment, pose)
            pose = np.matmul(frame_transform, pose)
            # convert to quaternion
            r = Rotation.from_matrix([pose[0][:3], pose[1][:3], pose[2][:3]])
            quaternion = r.as_quat()
            # write to file
            f.write(file.split('.')[0] + ' ')
            f.write(format(pose[0][-1], '.6f') + ' ')
            f.write(format(pose[1][-1], '.6f') + ' ')
            f.write(format(pose[2][-1], '.6f') + ' ')
            f.write(format(quaternion[0], '.6f') + ' ')
            f.write(format(quaternion[1], '.6f') + ' ')
            f.write(format(quaternion[2], '.6f') + ' ')
            f.write(format(quaternion[3], '.6f') + ' ')
            if file.split('.')[0] != str(len(files) - 1):
                f.write('\n')