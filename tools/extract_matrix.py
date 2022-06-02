### extract matrix from official scene****_**.txt 


import numpy as np
lines = open('/home/steve/dataset/scene0000_00/scene0000_00.txt').readlines()
axis_align_matrix = None
color2depth_matrix = None

for line in lines:
    if 'axisAlignment' in line:
        axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    if 'colorToDepthExtrinsics' in line:
        color2depth_matrix = [float(x) for x in line.rstrip().strip('colorToDepthExtrinsics = ').split(' ')]
   
axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
color2depth_matrix = np.array(color2depth_matrix).reshape((4, 4))
print(axis_align_matrix)
print('-------------------------------------------------')
print(color2depth_matrix)
print(np.linalg.inv(color2depth_matrix))
