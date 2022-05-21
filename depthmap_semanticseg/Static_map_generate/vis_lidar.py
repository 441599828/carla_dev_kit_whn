"""Open3D Lidar visuialization example for CARLA"""

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def main(arg):
    """Main function of the script"""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(2.0)
    world = client.get_world()

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.1

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        world.apply_settings(settings)

        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('dropoff_general_rate', str(0.0))
        lidar_bp.set_attribute('dropoff_intensity_limit', str(1.0))
        lidar_bp.set_attribute('dropoff_zero_intensity', str(0.0))
        lidar_bp.set_attribute('upper_fov', str(25))
        lidar_bp.set_attribute('lower_fov', str(-45))
        lidar_bp.set_attribute('channels', str(80))
        lidar_bp.set_attribute('range', str(200))
        lidar_bp.set_attribute('rotation_frequency', str(10))
        lidar_bp.set_attribute('points_per_second', str(1440000))

        lidar1_transform = carla.Transform(carla.Location(x=-32.2, y=30.0, z=6.0),
                                           carla.Rotation(pitch=-20.000, yaw=-180.0, roll=0.000000))
        lidar2_transform = carla.Transform(carla.Location(x=-49.2, y=43.2, z=6.5),
                                           carla.Rotation(pitch=-15.000, yaw=-90.0, roll=0.000000))
        lidar3_transform = carla.Transform(carla.Location(x=-52.2, y=21.2, z=6.5),
                                           carla.Rotation(pitch=-17.000, yaw=0.0, roll=0.000000))
        lidar4_transform = carla.Transform(carla.Location(x=-50.2, y=3.2, z=6.4),
                                           carla.Rotation(pitch=-18.000, yaw=-90.0, roll=0.000000))
        lidar5_transform = carla.Transform(carla.Location(x=106.2, y=41.2, z=6.3),
                                           carla.Rotation(pitch=-20.000, yaw=-125.0, roll=0.000000))
        lidar6_transform = carla.Transform(carla.Location(x=106.2, y=-2.2, z=6.2),
                                           carla.Rotation(pitch=-19.000, yaw=125.0, roll=0.000000))
        lidar7_transform = carla.Transform(carla.Location(x=57.2, y=69.2, z=6.5),
                                           carla.Rotation(pitch=-17.000, yaw=-150.0, roll=0.000000))
        lidar8_transform = carla.Transform(carla.Location(x=30.2, y=69.2, z=6.5),
                                           carla.Rotation(pitch=-18.000, yaw=-40.0, roll=0.000000))
        lidar9_transform = carla.Transform(carla.Location(x=44.2, y=37.2, z=6.2),
                                           carla.Rotation(pitch=-20.000, yaw=-40.0, roll=0.000000))
        lidar10_transform = carla.Transform(carla.Location(x=43.2, y=36.2, z=6.5),
                                            carla.Rotation(pitch=-20.000, yaw=-140.0, roll=0.000000))


        lidar = world.spawn_actor(blueprint=lidar_bp, transform=lidar10_transform)

        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: lidar_callback(data, point_list))

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        if arg.show_axis:
            add_open3d_axis(vis)

        frame = 0
        dt0 = datetime.now()
        while True:
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)

            vis.poll_events()
            vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(0.05)
            world.tick()

            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1
            # o3d.io.write_point_cloud(str(frame) + '.pcd', point_list, write_ascii=True)

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        lidar.destroy()
        vis.destroy_window()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
