"""
@author steve
@brief Generate TUM format ground truth trajectory from pose folder.
@details tx,ty,tz, qx, qy, qz, qr. Real part of quaternion is located in last one.
"""
import numpy as np
import os
from scipy.spatial.transform import Rotation

planefusion_format = True
scene_path = '/home/steve/dataset/scene0427_00_alignement'
pose_folder = os.path.join(scene_path, 'pose')

with open(os.path.join(scene_path, 'groundtruth.txt'), 'x') as f:
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
            r = Rotation.from_matrix([pose[0][:3], pose[1][:3], pose[2][:3]])
            quaternion = r.as_quat()
            # write to file
            f.write(file.split('.')[0] + ' ')
            f.write(str(pose[0][-1]) + ' ')
            f.write(str(pose[1][-1]) + ' ')
            f.write(str(pose[2][-1]) + ' ')
            f.write(str(quaternion[0]) + ' ')
            f.write(str(quaternion[1]) + ' ')
            f.write(str(quaternion[2]) + ' ')
            f.write(str(quaternion[3]))
            if file.split('.')[0] != str(len(files) - 1):
                f.write('\n')

