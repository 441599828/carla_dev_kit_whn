# image_downsample: both depth img and rgb img

import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':

    root = "../VirtualDataset"
    dst = "../GTAVDepth"
    cams = ["cam1", "cam2", "cam3", "cam4", "cam5"]
    for cam in cams:
        weatherlist = ["Foggy", "Overcast", "Rain", "Snow", "Sunny"]
        for weather in weatherlist:
            colorfile = os.path.join(root, cam, weather, "color")
            for color in tqdm(os.listdir(colorfile)):
                img_old = cv2.imread(os.path.join(colorfile, color))
                img_new = cv2.resize(img_old, (1920, 1080))
                if not os.path.exists(os.path.join(dst, "Images", cam, weather)):
                    os.makedirs(os.path.join(dst, "Images", cam, weather))
                cv2.imwrite(os.path.join(dst, "Images", cam, weather, cam + "_" + weather + "_" + color)[0:-4] + ".png",
                            img_new)

            depthfile = os.path.join(root, cam, weather, "depth")
            for depth in tqdm(os.listdir(depthfile)):
                imgraw = np.fromfile(os.path.join(depthfile, depth), dtype=np.float32)
                imgraw = imgraw.reshape(1440, 2560)
                b = 10003.814 * 0.15 / (-0.15 + 10003.814)
                k = 10003.814 / (-0.15 + 10003.814) - 1.0
                imgdepth = b / (imgraw + k)
                imgdepth = imgdepth * 256
                imgdepth = imgdepth.astype(np.uint16)
                imgdepth = cv2.resize(imgdepth, (1920, 1080))
                # index = imgdepth >= 150
                # imgdepth[index] = 0
                if not os.path.exists(os.path.join(dst, "Depth", cam, weather)):
                    os.makedirs(os.path.join(dst, "Depth", cam, weather))
                cv2.imwrite(os.path.join(dst, "Depth", cam, weather, cam + "_" + weather + "_" + depth)[0:-4] + ".png",
                            imgdepth)
                # test=np.array(Image.open(os.path.join(dst, "Depth", cam, weather, cam + "_" + weather + "_" + depth)[0:-4] + ".png"),dtype=int)
                # depth = test.astype(np.float) / 256.
                # depth[test == 0] = -1.
