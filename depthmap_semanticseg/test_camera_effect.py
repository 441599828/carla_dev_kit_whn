# test the image difference under different focal_distance
import glob
import os
import sys

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


def main():
    actor_list = []
    pygame.init()

    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    try:

        cam_rgb1 = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_rgb1.set_attribute("image_size_x", str(1024))
        cam_rgb1.set_attribute("image_size_y", str(576))
        cam_rgb1.set_attribute("fov", str(30))
        cam_rgb1.set_attribute("min_fstop", str(0.1))
        cam_rgb1.set_attribute("fstop", str(0.8))
        cam_rgb1.set_attribute("focal_distance", str(0.1))

        cam1_transform = carla.Transform(carla.Location(x=-27.0, y=29.0, z=5.3),
                                         carla.Rotation(pitch=0.000, yaw=-180.0, roll=0.000000))
        cam1_rgb = world.spawn_actor(cam_rgb1, cam1_transform)
        # print(cam1_rgb.attributes)
        # cam2_rgb = world.spawn_actor(cam_rgb2, cam1_transform)
        # cam3_rgb = world.spawn_actor(cam_rgb3, cam1_transform)
        actor_list.append(cam1_rgb)
        # actor_list.append(cam2_rgb)
        # actor_list.append(cam3_rgb)

        # Create a synchronous mode context.
        focal_dis = 400
        with CarlaSyncMode(world, cam1_rgb, fps=60) as sync_mode:
            for frame in range(0, 50):
                # while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, img1_rgb = sync_mode.tick(
                    timeout=2.0)
                # cam1_rgb.set_attribute("focal_distance", str(focal_dis))

                cam_rgb1.set_attribute("focal_distance", str(focal_dis))
                cam1_rgb = world.spawn_actor(cam_rgb1, cam1_transform)

                focal_dis *= 1.2
                img1_rgb.save_to_disk('rgb/%.6d_cam01rgb.jpg' % (focal_dis*10))
                # img2_rgb.save_to_disk('cam02rgb.jpg')
                # img3_rgb.save_to_disk('cam03rgb.jpg')
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
