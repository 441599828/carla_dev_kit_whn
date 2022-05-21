import os
import numpy as np
from tqdm import tqdm


def my_read_pcd(filepath):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    intensity = np.ones((points.shape[0], 1))
    points1 = np.concatenate((points, intensity), axis=1)
    return points1

def convert(pcdfolder, binfolder):
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    file_list = os.listdir(ori_path)
    des_path = os.path.join(current_path, binfolder)
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    for file in tqdm(file_list):
        (filename, extension) = os.path.splitext(file)
        velodyne_file = os.path.join(ori_path, filename) + '.pcd'
        pl = my_read_pcd(velodyne_file)
        pl = pl.reshape(-1, 4).astype(np.float32)
        velodyne_file_new = os.path.join(des_path, filename) + '.bin'
        pl.tofile(velodyne_file_new)


if __name__ == "__main__":
    convert('D:/IPS/fine_labeled_zhengshi/data_zhengshi/PCD_COM_ROI_IPU2coor/pcd', 'D:/IPS/fine_labeled_zhengshi/data_zhengshi/PCD_COM_ROI_IPU2coor/bin')