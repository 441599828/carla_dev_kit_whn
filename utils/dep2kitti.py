import os
from PIL import Image
import numpy as np


def dep_cvt(dep_file):
    dep = Image.open(dep_file)
    dep_array = np.array(dep)
    R = dep_array[:, :, 0]
    G = dep_array[:, :, 1]
    B = dep_array[:, :, 2]
    normalized = (R + G * 256 + B * 65536) / 16777215
    in_meters = (1000 * normalized) * 256
    depth = in_meters.astype(np.uint16)
    Image.fromarray(depth).save(dep_file.replace('dep/', 'depth/'))

    # test1 = Image.open('result.png')
    # test = np.array(test1, dtype=int)
    # depth1 = test.astype(np.float) / 256.


if __name__ == '__main__':
    tr_dep_pth = "./dep"
    tr_dep_files = []
    for root, dirs, files in os.walk(tr_dep_pth):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                dep_cvt(os.path.join(root, file))
