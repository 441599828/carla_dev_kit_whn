# 把cityscapes格式的instance segmentation png img转换成sctrainIDlabel.png
import numpy as np
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import os

def _vis(image):
    image.convert("L")
    img = np.array(image)
    plt.figure()
    plt.title("Train Image")
    plt.imshow(img)
    plt.show()

def process_one_frame(img_file, dst_file):
    img = Image.open(img_file)
    img = img.convert("I")
    img_array = np.array(img)

    mask_array = img_array//1000

    mask_trainid_array = deepcopy(mask_array)
    mask_trainid_array[mask_trainid_array==26] = 13
    mask_trainid_array[mask_trainid_array==24] = 11
    mask_trainid_array[mask_trainid_array==0] = 255

    mask = Image.fromarray(mask_trainid_array)
    mask = mask.convert("L")
    # _vis(mask)

    mask.save(dst_file)


if __name__ == "__main__":
    img_root = "C:/Users/wanghuanan/Desktop/ins_sc_format/cam10"
    dst_root = "C:/Users/wanghuanan/Desktop/trainIDlabel/cam10"
    
    img_files = [os.path.join(img_root, x) for x in os.listdir(img_root)]
    for img_file in img_files:
        dst_file = img_file.replace(img_root, dst_root)
        process_one_frame(img_file, dst_file)