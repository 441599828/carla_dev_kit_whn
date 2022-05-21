import os
import random


def randomsplit():
    Img_root = "../Data/Img"
    Dep_root = "./Carla/Dep"
    Seg_root = "./Carla/Seg"

    Img = []
    for root, dirs, files in os.walk(Img_root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                Img.append(os.path.join(root, file).replace(Img_root + '/', ''))
            else:
                print("None")
    Img.sort()

    Dep = []
    for root, dirs, files in os.walk(Dep_root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                Dep.append(os.path.join(root, file).replace(Dep_root + '/', ''))
    Dep.sort()

    Seg = []
    for root, dirs, files in os.walk(Seg_root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                Dep.append(os.path.join(root, file).replace(Dep_root + '/', ''))
    Seg.sort()

    assert len(Img) == len(Dep)
    assert len(Img) == len(Seg)

    Image_Depth = []
    for i in range(len(Img)):
        Image_Depth.append(Img[i] + ' ' + Dep[i])

    # random split data in train:val:test=6:2:2
    random.shuffle(Image_Depth)
    train = Image_Depth[0:int(2 / 3 * len(Image_Depth))]
    val = Image_Depth[int(2 / 3 * len(Image_Depth)):int(5 / 6 * len(Image_Depth))]
    test = Image_Depth[int(5 / 6 * len(Image_Depth)):]

    f = open("GTAVDepth_train_files_with_gt.txt", "w")
    splitstr = '\n'
    # f.write(splitstr.join(train))
    f.write(splitstr.join(train[0:int(len(train) / 4)]))
    f.close()

    f = open("GTAVDepth_val_files_with_gt.txt", "w")
    # f.write(splitstr.join(val))
    f.write(splitstr.join(val[0:int(len(val) / 4)]))
    f.close()

    f = open("GTAVDepth_test_files_with_gt.txt", "w")
    # f.write(splitstr.join(test))
    f.write(splitstr.join(test[0:int(len(test) / 4)]))
    f.close()


def train234579valtest16810():
    Img_root = "../Data/rgb"
    Dep_root = "../Data/depth"
    Seg_root = "../Data/seg"

    Img = []
    for root, dirs, files in os.walk(Img_root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                Img.append(os.path.join(root, file).replace(Img_root + '/', ''))
            else:
                print("None")
    Img.sort()

    Dep = []
    for root, dirs, files in os.walk(Dep_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                Dep.append(os.path.join(root, file).replace(Dep_root + '/', ''))
    Dep.sort()

    Seg = []
    for root, dirs, files in os.walk(Seg_root):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                Seg.append(os.path.join(root, file).replace(Seg_root + '/', ''))
    Seg.sort()

    assert len(Img) == len(Dep)
    assert len(Img) == len(Seg)

    Img_Dep_Seg = []
    for i in range(len(Img)):
        Img_Dep_Seg.append(Img[i] + ' ' + Dep[i] + ' ' + Seg[i])

    # random split data in train:val:test=6:2:2
    train_cam = ['cam02', 'cam03', 'cam04', 'cam05', 'cam07', 'cam09']
    train = []
    for x in train_cam:
        for filename in Img_Dep_Seg:
            if x in filename:
                train.append(filename)
    val_test_cam = ['cam01', 'cam06', 'cam08', 'cam10']
    val_test = []
    for x in val_test_cam:
        for filename in Img_Dep_Seg:
            if x in filename:
                val_test.append(filename)
    random.shuffle(val_test)
    val = val_test[0:int(1 / 2 * len(val_test))]
    test = val_test[int(1 / 2 * len(val_test)):]
    # split train_val_test in folder
    from shutil import copy
    from tqdm import tqdm
    for i in tqdm(train):
        [rgb, depth, seg] = i.split(' ')
        rgb = os.path.join(Img_root, rgb)
        depth = os.path.join(Dep_root, depth)
        seg = os.path.join(Seg_root, seg)
        copy(rgb, rgb.replace(Img_root, "../Data/train/rgb"))
        copy(depth, depth.replace(Dep_root, "../Data/train/depth"))
        copy(seg, seg.replace(Seg_root, "../Data/train/seg"))
    for i in tqdm(val):
        [rgb, depth, seg] = i.split(' ')
        rgb = os.path.join(Img_root, rgb)
        depth = os.path.join(Dep_root, depth)
        seg = os.path.join(Seg_root, seg)
        copy(rgb, rgb.replace(Img_root, "../Data/val/rgb"))
        copy(depth, depth.replace(Dep_root, "../Data/val/depth"))
        copy(seg, seg.replace(Seg_root, "../Data/val/seg"))
    for i in tqdm(test):
        [rgb, depth, seg] = i.split(' ')
        rgb = os.path.join(Img_root, rgb)
        depth = os.path.join(Dep_root, depth)
        seg = os.path.join(Seg_root, seg)
        copy(rgb, rgb.replace(Img_root, "../Data/test_middle/rgb"))
        copy(depth, depth.replace(Dep_root, "../Data/test_middle/depth"))
        copy(seg, seg.replace(Seg_root, "../Data/test_middle/seg"))

    f = open("Carla_train_files_with_gt.txt", "w")
    splitstr = '\n'
    f.write(splitstr.join(train))
    f.close()

    f = open("Carla_val_files_with_gt.txt", "w")
    f.write(splitstr.join(val))
    f.close()

    f = open("Carla_test_filesmiddle_with_gt.txt", "w")
    f.write(splitstr.join(test))
    f.close()


def add_sebfolder_txt_file():
    test_easy_rgb = os.listdir("../Data/test_hardest/rgb")
    test_easy = []
    for i in test_easy_rgb:
        test_easy.append(i + ' ' + i.replace('rgb.jpg', 'dep.png') + ' ' + i.replace('rgb.jpg', 'seg.jpg'))
    f = open("/media/whn/新加卷/dataset/carla/Data/Carla_testhardest_files_with_gt.txt", "w")
    splitstr = '\n'
    f.write(splitstr.join(test_easy))
    f.close()
    # test_hard_rgb = os.listdir("../Data/test_hard/rgb")
    # test_hard = []
    # for i in test_hard_rgb:
    #     test_hard.append(i + ' ' + i.replace('rgb.jpg', 'dep.png') + ' ' + i.replace('rgb.jpg', 'seg.jpg'))
    # f = open("/media/whn/新加卷/dataset/carla/Data/Carla_testhard_files_with_gt.txt", "w")
    # splitstr = '\n'
    # f.write(splitstr.join(test_hard))
    # f.close()


if __name__ == "__main__":
    # train234579valtest16810()
    add_sebfolder_txt_file()
