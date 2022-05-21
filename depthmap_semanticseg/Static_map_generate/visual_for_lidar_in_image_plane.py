import glob
import os
import sys
import numpy as np
from matplotlib import cm
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame

try:
    import queue
except ImportError:
    import Queue as queue

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds
        ))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main(args):
    actor_list = []
    pygame.init()

    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    try:

        # Search the desired blueprints
        camera_bp = world.get_blueprint_library().filter("sensor.camera.rgb")[0]
        lidar_bp = world.get_blueprint_library().filter("sensor.lidar.ray_cast")[0]
        # Configure the blueprints
        camera_bp.set_attribute("image_size_x", str(1024))
        camera_bp.set_attribute("image_size_y", str(576))
        camera_bp.set_attribute("fov", str(105))
        cam1_transform = carla.Transform(carla.Location(x=-32.0, y=28.8, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-180.0, roll=0.000000))
        cam2_transform = carla.Transform(carla.Location(x=-49.0, y=43.0, z=6.3),
                                         carla.Rotation(pitch=-35.000, yaw=-90.0, roll=0.000000))
        cam3_transform = carla.Transform(carla.Location(x=-52.0, y=21.0, z=6.2),
                                         carla.Rotation(pitch=-37.000, yaw=0.0, roll=0.000000))
        cam4_transform = carla.Transform(carla.Location(x=-50.0, y=3.0, z=6.1),
                                         carla.Rotation(pitch=-38.000, yaw=-90.0, roll=0.000000))
        cam5_transform = carla.Transform(carla.Location(x=106.0, y=41.0, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-125.0, roll=0.000000))
        cam6_transform = carla.Transform(carla.Location(x=106.0, y=-2.0, z=6.4),
                                         carla.Rotation(pitch=-39.000, yaw=125.0, roll=0.000000))
        cam7_transform = carla.Transform(carla.Location(x=57, y=69.0, z=6.2),
                                         carla.Rotation(pitch=-37.000, yaw=-150.0, roll=0.000000))
        cam8_transform = carla.Transform(carla.Location(x=30, y=69.0, z=6.2),
                                         carla.Rotation(pitch=-38.000, yaw=-40.0, roll=0.000000))
        cam9_transform = carla.Transform(carla.Location(x=44.0, y=37.0, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-40.0, roll=0.000000))
        cam10_transform = carla.Transform(carla.Location(x=43.0, y=36.0, z=6.5),
                                          carla.Rotation(pitch=-40.000, yaw=-140.0, roll=0.000000))

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

        # Spawn the blueprints
        camera = world.spawn_actor(blueprint=camera_bp, transform=cam10_transform)
        lidar = world.spawn_actor(blueprint=lidar_bp, transform=lidar10_transform)
        actor_list.append(lidar)
        actor_list.append(camera)
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0
        # print(K)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera, lidar, fps=10) as sync_mode:
            for frame in range(0, 1):
                # while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_data, lidar_data = sync_mode.tick(
                    timeout=2.0)

                # Get the raw BGRA buffer and convert it to an array of RGB of
                # shape (image_data.height, image_data.width, 3).
                im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
                im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
                im_array = im_array[:, :, :3][:, :, ::-1]

                image = Image.fromarray(im_array)
                image.save("%08drgb.png" % image_data.frame)

                # Get the lidar data and convert it to a numpy array.
                p_cloud_size = len(lidar_data)
                p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
                p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

                # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
                # focus on the 3D points.
                intensity = np.array(p_cloud[:, 3])

                # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
                local_lidar_points = np.array(p_cloud[:, :3]).T

                # Add an extra 1.0 at the end of each 3d point so it becomes of
                # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
                local_lidar_points = np.r_[
                    local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

                # This (4, 4) matrix transforms the points from lidar space to world space.
                lidar_2_world = lidar.get_transform().get_matrix()

                # Transform the points from lidar space to world space.
                world_points = np.dot(lidar_2_world, local_lidar_points)

                # This (4, 4) matrix transforms the points from world to sensor coordinates.
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

                # Transform the points from world space to camera space.
                sensor_points = np.dot(world_2_camera, world_points)

                # New we must change from UE4's coordinate system to an "standard"
                # camera coordinate system (the same used by OpenCV):

                # ^ z                       . z
                # |                        /
                # |              to:      +-------> x
                # | . x                   |
                # |/                      |
                # +-------> y             v y

                # This can be achieved by multiplying by the following matrix:
                # [[ 0,  1,  0 ],
                #  [ 0,  0, -1 ],
                #  [ 1,  0,  0 ]]

                # Or, in this case, is the same as swapping:
                # (x, y ,z) -> (y, -z, x)
                point_in_camera_coords = np.array([
                    sensor_points[1],
                    sensor_points[2] * -1,
                    sensor_points[0]])

                # Finally we can use our K matrix to do the actual 3D -> 2D.
                points_2d = np.dot(K, point_in_camera_coords)

                # Remember to normalize the x, y values by the 3rd value.
                points_2d = np.array([
                    points_2d[0, :] / points_2d[2, :],
                    points_2d[1, :] / points_2d[2, :],
                    points_2d[2, :]])

                # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
                # contains all the y values of our points. In order to properly
                # visualize everything on a screen, the points that are out of the screen
                # must be discarted, the same with points behind the camera projection plane.
                points_2d = points_2d.T
                intensity = intensity.T
                points_in_canvas_mask = \
                    (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
                    (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
                    (points_2d[:, 2] > 0.0)
                points_2d = points_2d[points_in_canvas_mask]
                intensity = intensity[points_in_canvas_mask]

                # Extract the screen coords (uv) as integers.
                u_coord = points_2d[:, 0].astype(np.int)
                v_coord = points_2d[:, 1].astype(np.int)

                # Since at the time of the creation of this script, the intensity function
                # is returning high values, these are adjusted to be nicely visualized.
                intensity = 4 * intensity - 3
                color_map = np.array([
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

                if args.dot_extent <= 0:
                    # Draw the 2d points on the image as a single pixel using numpy.
                    im_array[v_coord, u_coord] = color_map
                else:
                    # Draw the 2d points on the image as squares of extent args.dot_extent.
                    for i in range(len(points_2d)):
                        # I'm not a NumPy expert and I don't know how to set bigger dots
                        # without using this loop, so if anyone has a better solution,
                        # make sure to update this script. Meanwhile, it's fast enough :)
                        im_array[
                        v_coord[i] - args.dot_extent: v_coord[i] + args.dot_extent,
                        u_coord[i] - args.dot_extent: u_coord[i] + args.dot_extent] = color_map[i]

                # # Save the image using Pillow module.
                # image = Image.fromarray(im_array)
                # image.save("%08drgbvslidar.png" % image_data.frame)
                
                # import open3d as o3d
                print(K)
                # # lidar_2_camera = np.dot(world_2_camera, lidar_2_world)
                # # print(lidar_2_camera)
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(p_cloud[:, :3])
                # o3d.io.write_point_cloud(str(frame) + '.pcd', pcd, write_ascii=True)
                #
                img = np.zeros([image_h, image_w], dtype=np.uint16)
                z = points_2d[:, 2]
                for i in range(0, u_coord.size):
                    img[v_coord[i]][u_coord[i]] = z[i] * 256
                Image.fromarray(img).save(str(frame) + '.png')

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        import argparse

        argparser = argparse.ArgumentParser(description='CARLA Sensor sync and projection tutorial')

        argparser.add_argument(
            '-d', '--dot-extent',
            metavar='SIZE',
            default=2,
            type=int,
            help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
        args = argparser.parse_args()
        args.dot_extent -= 1
        main(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
