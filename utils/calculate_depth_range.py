# calculate depth ranage from depth img for collected data
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def calcu_minmax_depth(tr_dep_pth, tr_dep_files):
    max_dep, min_dep = -1, 5000
    for dep_file in tqdm(tr_dep_files, desc="calculating minmax depth", mininterval=0.1):
        dep_file = os.path.join(tr_dep_pth, dep_file)
        test = np.array(Image.open(dep_file), dtype=int)
        depth = test.astype(np.float) / 256.
        if np.max(depth) >= max_dep:
            max_dep = np.max(depth)
        if np.min(depth) <= min_dep:
            min_dep = np.min(depth)
    print("Max depth is: ", max_dep)
    print("Min depth is: ", min_dep)

if __name__=='__main__':

    tr_dep_pth = "../Data"
    tr_dep_files = []
    for root, dirs, files in os.walk(tr_dep_pth):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                tr_dep_files.append(os.path.join(root, file).replace(tr_dep_pth + '/', ''))
    tr_dep_files.sort()

    calcu_minmax_depth(tr_dep_pth, tr_dep_files)