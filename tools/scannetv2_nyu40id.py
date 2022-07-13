import glob as gb
import sys
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import re



# nyu40d
nyu40_classes = ['unlabeled', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
   'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 
   'curtain', 'dresser', 'pillow', 'mirror', 'floormat', 'clothes', 'ceiling', 'books', 
   'refrigerator', 'television', 'paper', 'towel', 'showercurtain', 'box', 'whiteboard', 
   'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 
   'otherfurniture', 'otherprop']

nyu40_colormap = [(0, 0, 0), (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), 
(188, 189, 34), (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), 
(148, 103, 189), (196, 156, 148), (23, 190, 207), (178, 76, 76), (247, 182, 210), 
(66, 188, 102), (219, 219, 141), (140, 57, 197), (202, 185, 52), (51, 176, 203), 
(200, 54, 131), (92, 193, 61), (78, 71, 183), (172, 114, 82), (255, 127, 14), 
(91, 163, 138), (153, 98, 156), (140, 153, 101), (158, 218, 229), (100, 125, 154), 
(178, 127, 135), (120, 185, 128), (146, 111, 194), (44, 160, 44), (112, 128, 144), 
(96, 207, 209), (227, 119, 194), (213, 92, 176), (94, 106, 211), (82, 84, 163), 
(100, 85, 144)]




# resize adjust to the depth image size
rdim = (640, 480)

# utils function
def sort_key(s):
    # sort by 0,1,2,3...
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)

# utils function
def strsort(alist):
    alist.sort(key=sort_key)
    return alist


# Get nyu40 class name list from my stored csv file
def get_nyu40_id_names(my_label_path = './scannet_info/mylabel.csv'):
    """get the nyu40_id_names
    Returns: list --index: int, nyu40id
                  --elements: str, nyu40idname
    """
    lines = [line.rstrip() for line in open(my_label_path)]
    nyu40_id_names = []
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        nyu40_id_names.append(elements[1])
    nyu40_id_names = nyu40_id_names[1:]
    return nyu40_id_names

def cvt_scan2nyu40(scannet_tsv = './scannet_info/scannetv2-labels.combined.tsv'):
    """convert scannetv2 id value to nyu40 id
    Returns: dic --- index: str, i.e. str(scannetv2 id value)
                 --- element: str, i.e. str(nyu40 id value)
             list --- index: int, nyu40 id value
                  --- element: int, the number of every class
    """
    lines = [line.rstrip() for line in open(scannet_tsv)]
    lines = lines[1:]
    scan2nyu = {}
    nyu40_class_num = [0]*41
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        scan2nyu[elements[0]] = elements[4]
        nyu40_class_num[int(elements[4])] = nyu40_class_num[int(elements[4])] + int(elements[3])
    return scan2nyu, nyu40_class_num 

if __name__ == '__main__':
    scan_nyu40_map, dis = cvt_scan2nyu40()
    print(f'Id map between ScanNet and nyu40 {scan_nyu40_map}')
    # print(dis)

    # raw label image -- 16 bit
    raw_img = cv2.imread('scannet_info/test_images/6.png', -1)
    img = cv2.resize(raw_img, rdim, interpolation = cv2.INTER_NEAREST)
    print(img.shape)

    rows, cols = img.shape[0], img.shape[1]
    """
    convert raw to nyu40id image
    """
    for i in range(rows):
        for j in range(cols):
            index = str(img[i][j])
            if index == '0':
                # unlabelde
                continue
            img[i][j] = scan_nyu40_map[index]
    # if (cv2.imwrite('4.png', img)):
    #     print("Write Successfully ")
    # else
    #     with open('imgFailedName.txt', 'a', encoding='utf-8') as f:
    #         f.write(mylabeldir + "failed\n")

    """
    Visualize nyu40 lable image 
    """
    colorimg = np.zeros((rows, cols, 3), dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            colorimg[i, j, 0] = nyu40_colormap[img[i,j]][2] # B
            colorimg[i, j, 1] = nyu40_colormap[img[i,j]][1] # G
            colorimg[i, j, 2] = nyu40_colormap[img[i,j]][0] # R
    
 
    wall_nyu40id = nyu40_classes.index('wall')
    floor_nyu40id = nyu40_classes.index('floor')
    ceiling_nyu40id = nyu40_classes.index('ceiling')

    print(f"wall nyu40 id:color, {wall_nyu40id}: {nyu40_colormap[wall_nyu40id]}")
    print(f"floor nyu40 id:color, {floor_nyu40id}: {nyu40_colormap[floor_nyu40id]}")
    print(f"ceiling nyu40 id:color, {ceiling_nyu40id}: {nyu40_colormap[ceiling_nyu40id]}")

    cv2.imshow('color', colorimg)
    cv2.waitKey()
    cv2.destroyAllWindows()

