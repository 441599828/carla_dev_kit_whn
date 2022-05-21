import numpy as np
import os
import PIL.Image as Image
import open3d as o3d


def velo2img(velofile, calibfile, imgfile):
    # read from file
    img = np.array(Image.open(imgfile))
    image_h, image_w = img.shape[0], img.shape[1]

    scan = o3d.io.read_point_cloud(velofile)
    points = np.asarray(scan.points).T
    local_lidar_points = np.r_[points, [np.ones(points.shape[1])]]

    with open(calibfile, 'r') as f:
        calib = f.readlines()
    K = np.array(calib[0].split()[1:]).astype(np.float).reshape((3, 3))
    lidar_2_camera = np.array(calib[1].split()[1:]).astype(np.float).reshape((4, 4))

    # Transform the points from world space to camera space.
    sensor_points = np.dot(lidar_2_camera, local_lidar_points)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])
    points_2d = np.dot(K, point_in_camera_coords)
    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    points_2d = points_2d.T
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    points_2d = points_2d[points_in_canvas_mask]
    return points_2d


def save_depth_png(depth_root, depth):
    IMG_H, IMG_W = 576, 1024
    img = np.zeros([IMG_H, IMG_W], dtype=np.uint16)
    w, h, z = depth
    h = np.floor(h).astype(np.uint).flatten()
    w = np.floor(w).astype(np.uint).flatten()
    z = z.flatten()
    for i in range(0, h.size):
        img[h[i]][w[i]] = z[i] * 256
    img = img.astype(np.uint16)
    Image.fromarray(img).save(depth_root)


if __name__ == '__main__':
    velo_files = sorted(os.listdir("../../static_map/from_lidar/pcd"))
    calib_files = sorted(os.listdir("../../static_map/calib"))
    img_files = sorted(os.listdir('../../static_map/from_camera/rgb'))

    depth_root = "../../static_map/from_lidar/depth"

    for i in range(0, len(velo_files)):
        depth = velo2img(os.path.join("../../static_map/from_lidar/pcd", velo_files[i]),
                         os.path.join("../../static_map/calib", calib_files[i]),
                         os.path.join('../../static_map/from_camera/rgb', img_files[i]))
        depth_name = velo_files[i].replace('.pcd', '.png')
        save_depth_png(os.path.join(depth_root, depth_name), depth.T)
        print('Saved proj depth: ', depth_name)
