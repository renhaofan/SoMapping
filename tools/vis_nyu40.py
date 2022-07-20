import glob as gb
import sys
import cv2
import os
from cv2 import COLORMAP_SPRING
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


img = cv2.imread('340.png', -1)

rows, cols = img.shape[0], img.shape[1]
colorimg = np.zeros((rows, cols, 3), dtype='uint8')
for i in range(rows):
    for j in range(cols):
        colorimg[i, j, 0] = nyu40_colormap[img[i,j]][2] # B
        colorimg[i, j, 1] = nyu40_colormap[img[i,j]][1] # G
        colorimg[i, j, 2] = nyu40_colormap[img[i,j]][0] # R

while True:
    cv2.imshow('color', colorimg)
    k = cv2.waitKey()
    if k == 27:
        break

cv2.destroyAllWindows()