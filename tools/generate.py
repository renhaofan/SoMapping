"""

@author: steve
@description: Generate TUM format camera concerned txt file from
ScanNet scene****_**.txt
@note: Not sure how many significant digits are reserved is reasonable
after inverse operation. Temporarily retain 6 digits, consistent with
the original scene****_**.txt file.

"""
import numpy as np

txt_path = '/home/steve/dataset/scene0000_00/scene0000_00.txt'
save_name = txt_path.split('/')[-1].split('.')[0] + '_TUM_Format.txt'

valid_bitnum = 6

# Color camera concerned
color_w = None
color_h = None
color_fx = None
color_fy = None
color_cx = None
color_cy = None

# Depth camera concerned
depth_w = None
depth_h = None
color_fx = None
color_fy = None
color_cx = None
color_cy = None

# Translation value for normalize scene.
axis_align_matrix = None
# Extrinsics from color camera to depth camera.
color2depth_matrix = None

# Read ScanNet scene file
with open(txt_path, 'r') as f:
    lines = f.readlines()
    # Read one line every time
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
            continue
        if 'colorToDepthExtrinsics' in line:
            color2depth_matrix = [float(x) for x in line.rstrip().strip('colorToDepthExtrinsics = ').split(' ')]
            color2depth_matrix = np.array(color2depth_matrix).reshape((4, 4))
            continue
        if 'colorHeight' in line:
            color_h = int(line.rstrip().strip('colorHeight = '))
            continue
        if 'colorWidth' in line:
            color_w = int(line.rstrip().strip('colorWidth = '))
            continue
        if 'depthHeight' in line:
            depth_h = int(line.rstrip().strip('depthHeight = '))
            continue
        if 'depthWidth' in line:
            depth_w = int(line.rstrip().strip('depthWidth = '))
            continue
        if 'fx_color' in line:
            color_fx = float(line.rstrip().strip('fx_color = '))
            continue
        if 'fy_color' in line:
            color_fy = float(line.rstrip().strip('fy_color = '))
            continue
        if 'mx_color' in line:
            color_cx = float(line.rstrip().strip('mx_color = '))
            continue
        if 'my_color' in line:
            color_cy = float(line.rstrip().strip('my_color = '))
            continue
        if 'fx_depth' in line:
            depth_fx = float(line.rstrip().strip('fx_depth = '))
            continue
        if 'fy_depth' in line:
            depth_fy = float(line.rstrip().strip('fy_depth = '))
            continue
        if 'mx_depth' in line:
            depth_cx = float(line.rstrip().strip('mx_depth = '))
            continue
        if 'my_depth' in line:
            depth_cy = float(line.rstrip().strip('my_depth = '))
            continue

depth2color_matrix = np.linalg.inv(color2depth_matrix)

with open(save_name, 'w') as f:
    f.write(str(color_w) + ' ' + str(color_h) + '\n')
    f.write(format(color_fx, '.6f') + ' ' + format(color_fy, '.6f') + '\n')
    f.write(format(color_cx, '.6f') + ' ' + format(color_cy, '.6f') + '\n')
    f.write('\n')
    f.write(str(depth_w) + ' ' + str(depth_h) + '\n')
    f.write(format(depth_fx, '.6f') + ' ' + format(depth_fy, '.6f') + '\n')
    f.write(format(depth_cx, '.6f') + ' ' + format(depth_cy, '.6f') + '\n')

    f.write('\n')
    for line in depth2color_matrix[:3,:]:
        f.write(format(line[0], '.6f') + ' ' + format(line[1], '.6f') + ' ' + \
                format(line[2], '.6f') + ' ' + format(line[3], '.6f') + '\n')
    f.write('\n')
    f.write('1.0 0.0 0.0 0.0' + '\n') 
    f.write('0.0 1.0 0.0 0.0' + '\n') 
    f.write('0.0 0.0 1.0 0.0' + '\n') 
    f.write('\n')
    f.write('affine 0.001 0.0')