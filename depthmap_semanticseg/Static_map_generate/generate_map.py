# if u want to change data amount, change data_collect.py(line 240,line 249) and set_weather.py both.
# terminal 1 :bash CarlaUE4.sh
# terminal 2 :python generate_traffic.py -n 50 -w 50 --safe --generationw=All
# terminal 3 :python set_weather.py
# terminal 4 :python carla_test3.py

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


def set_cam(world, img_width, img_height, fov):
    # Spawn attached RGB camera
    cam_rgb = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_rgb.set_attribute("image_size_x", str(img_width))
    cam_rgb.set_attribute("image_size_y", str(img_height))
    cam_rgb.set_attribute("fov", str(fov))

    # Spawn attached Depth camera
    cam_dep = world.get_blueprint_library().find('sensor.camera.depth')
    cam_dep.set_attribute("image_size_x", str(img_width))
    cam_dep.set_attribute("image_size_y", str(img_height))
    cam_dep.set_attribute("fov", str(fov))

    # Spawn attached semantic_segmentation camera
    cam_seg = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    cam_seg.set_attribute("image_size_x", str(img_width))
    cam_seg.set_attribute("image_size_y", str(img_height))
    cam_seg.set_attribute("fov", str(fov))

    return [cam_rgb, cam_dep, cam_seg]


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

        [cam_rgb, cam_dep, cam_seg] = set_cam(world, img_width=1024, img_height=576, fov=105)
        # # Build the K projection matrix:
        # # K = [[Fx,  0, image_w/2],
        # #      [ 0, Fy, image_h/2],
        # #      [ 0,  0,         1]]
        # image_w = camera_bp.get_attribute("image_size_x").as_int()
        # image_h = camera_bp.get_attribute("image_size_y").as_int()
        # fov = camera_bp.get_attribute("fov").as_float()
        # focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        #
        # # In this case Fx and Fy are the same since the pixel aspect
        # # ratio is 1
        # K = np.identity(3)
        # K[0, 0] = K[1, 1] = focal
        # K[0, 2] = image_w / 2.0
        # K[1, 2] = image_h / 2.0

        cam1_transform = carla.Transform(carla.Location(x=-32.0, y=28.8, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-180.0, roll=0.000000))
        cam1_rgb = world.spawn_actor(cam_rgb, cam1_transform)
        actor_list.append(cam1_rgb)
        cam1_dep = world.spawn_actor(cam_dep, cam1_transform)
        actor_list.append(cam1_dep)
        cam1_seg = world.spawn_actor(cam_seg, cam1_transform)
        actor_list.append(cam1_seg)

        cam2_transform = carla.Transform(carla.Location(x=-49.0, y=43.0, z=6.3),
                                         carla.Rotation(pitch=-35.000, yaw=-90.0, roll=0.000000))
        cam2_rgb = world.spawn_actor(cam_rgb, cam2_transform)
        actor_list.append(cam2_rgb)
        cam2_dep = world.spawn_actor(cam_dep, cam2_transform)
        actor_list.append(cam2_dep)
        cam2_seg = world.spawn_actor(cam_seg, cam2_transform)
        actor_list.append(cam2_seg)

        cam3_transform = carla.Transform(carla.Location(x=-52.0, y=21.0, z=6.2),
                                         carla.Rotation(pitch=-37.000, yaw=0.0, roll=0.000000))
        cam3_rgb = world.spawn_actor(cam_rgb, cam3_transform)
        actor_list.append(cam3_rgb)
        cam3_dep = world.spawn_actor(cam_dep, cam3_transform)
        actor_list.append(cam3_dep)
        cam3_seg = world.spawn_actor(cam_seg, cam3_transform)
        actor_list.append(cam3_seg)

        cam4_transform = carla.Transform(carla.Location(x=-50.0, y=3.0, z=6.1),
                                         carla.Rotation(pitch=-38.000, yaw=-90.0, roll=0.000000))
        cam4_rgb = world.spawn_actor(cam_rgb, cam4_transform)
        actor_list.append(cam4_rgb)
        cam4_dep = world.spawn_actor(cam_dep, cam4_transform)
        actor_list.append(cam4_dep)
        cam4_seg = world.spawn_actor(cam_seg, cam4_transform)
        actor_list.append(cam4_seg)

        cam5_transform = carla.Transform(carla.Location(x=106.0, y=41.0, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-125.0, roll=0.000000))
        cam5_rgb = world.spawn_actor(cam_rgb, cam5_transform)
        actor_list.append(cam5_rgb)
        cam5_dep = world.spawn_actor(cam_dep, cam5_transform)
        actor_list.append(cam5_dep)
        cam5_seg = world.spawn_actor(cam_seg, cam5_transform)
        actor_list.append(cam5_seg)

        cam6_transform = carla.Transform(carla.Location(x=106.0, y=-2.0, z=6.4),
                                         carla.Rotation(pitch=-39.000, yaw=125.0, roll=0.000000))
        cam6_rgb = world.spawn_actor(cam_rgb, cam6_transform)
        actor_list.append(cam6_rgb)
        cam6_dep = world.spawn_actor(cam_dep, cam6_transform)
        actor_list.append(cam6_dep)
        cam6_seg = world.spawn_actor(cam_seg, cam6_transform)
        actor_list.append(cam6_seg)

        cam7_transform = carla.Transform(carla.Location(x=57, y=69.0, z=6.2),
                                         carla.Rotation(pitch=-37.000, yaw=-150.0, roll=0.000000))
        cam7_rgb = world.spawn_actor(cam_rgb, cam7_transform)
        actor_list.append(cam7_rgb)
        cam7_dep = world.spawn_actor(cam_dep, cam7_transform)
        actor_list.append(cam7_dep)
        cam7_seg = world.spawn_actor(cam_seg, cam7_transform)
        actor_list.append(cam7_seg)

        cam8_transform = carla.Transform(carla.Location(x=30, y=69.0, z=6.2),
                                         carla.Rotation(pitch=-38.000, yaw=-40.0, roll=0.000000))
        cam8_rgb = world.spawn_actor(cam_rgb, cam8_transform)
        actor_list.append(cam8_rgb)
        cam8_dep = world.spawn_actor(cam_dep, cam8_transform)
        actor_list.append(cam8_dep)
        cam8_seg = world.spawn_actor(cam_seg, cam8_transform)
        actor_list.append(cam8_seg)

        cam9_transform = carla.Transform(carla.Location(x=44.0, y=37.0, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-40.0, roll=0.000000))
        cam9_rgb = world.spawn_actor(cam_rgb, cam9_transform)
        actor_list.append(cam9_rgb)
        cam9_dep = world.spawn_actor(cam_dep, cam9_transform)
        actor_list.append(cam9_dep)
        cam9_seg = world.spawn_actor(cam_seg, cam9_transform)
        actor_list.append(cam9_seg)

        cam10_transform = carla.Transform(carla.Location(x=43.0, y=36.0, z=6.5),
                                          carla.Rotation(pitch=-40.000, yaw=-140.0, roll=0.000000))
        cam10_rgb = world.spawn_actor(cam_rgb, cam10_transform)
        actor_list.append(cam10_rgb)
        cam10_dep = world.spawn_actor(cam_dep, cam10_transform)
        actor_list.append(cam10_dep)
        cam10_seg = world.spawn_actor(cam_seg, cam10_transform)
        actor_list.append(cam10_seg)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, cam1_rgb, cam1_dep, cam1_seg, cam2_rgb, cam2_dep, cam2_seg, cam3_rgb, cam3_dep,
                           cam3_seg, cam4_rgb, cam4_dep, cam4_seg,
                           cam5_rgb, cam5_dep, cam5_seg, cam6_rgb, cam6_dep, cam6_seg, cam7_rgb, cam7_dep, cam7_seg,
                           cam8_rgb, cam8_dep, cam8_seg, cam9_rgb,
                           cam9_dep, cam9_seg, cam10_rgb, cam10_dep, cam10_seg,
                           fps=40) as sync_mode:
            for frame in range(0, 1):
                # while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, img1_rgb, img1_dep, img1_seg, img2_rgb, img2_dep, img2_seg, img3_rgb, img3_dep, img3_seg, img4_rgb, img4_dep, img4_seg, img5_rgb, img5_dep, img5_seg, img6_rgb, img6_dep, img6_seg, img7_rgb, img7_dep, img7_seg, img8_rgb, img8_dep, img8_seg, img9_rgb, img9_dep, img9_seg, img10_rgb, img10_dep, img10_seg = sync_mode.tick(
                    timeout=2.0)
                img1_rgb.save_to_disk('rgb/%.6d_cam01rgb.jpg' % img1_rgb.frame)
                img1_dep.save_to_disk('dep/%.6d_cam01dep.png' % img1_dep.frame, carla.ColorConverter.Raw)
                img1_seg.save_to_disk('seg/%.6d_cam01seg.jpg' % img1_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img2_rgb.save_to_disk('rgb/%.6d_cam02rgb.jpg' % img2_rgb.frame)
                img2_dep.save_to_disk('dep/%.6d_cam02dep.png' % img2_dep.frame, carla.ColorConverter.Raw)
                img2_seg.save_to_disk('seg/%.6d_cam02seg.jpg' % img2_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img3_rgb.save_to_disk('rgb/%.6d_cam03rgb.jpg' % img3_rgb.frame)
                img3_dep.save_to_disk('dep/%.6d_cam03dep.png' % img3_dep.frame, carla.ColorConverter.Raw)
                img3_seg.save_to_disk('seg/%.6d_cam03seg.jpg' % img3_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img4_rgb.save_to_disk('rgb/%.6d_cam04rgb.jpg' % img4_rgb.frame)
                img4_dep.save_to_disk('dep/%.6d_cam04dep.png' % img4_dep.frame, carla.ColorConverter.Raw)
                img4_seg.save_to_disk('seg/%.6d_cam04seg.jpg' % img4_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img5_rgb.save_to_disk('rgb/%.6d_cam05rgb.jpg' % img5_rgb.frame)
                img5_dep.save_to_disk('dep/%.6d_cam05dep.png' % img5_dep.frame, carla.ColorConverter.Raw)
                img5_seg.save_to_disk('seg/%.6d_cam05seg.jpg' % img5_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img6_rgb.save_to_disk('rgb/%.6d_cam06rgb.jpg' % img6_rgb.frame)
                img6_dep.save_to_disk('dep/%.6d_cam06dep.png' % img6_dep.frame, carla.ColorConverter.Raw)
                img6_seg.save_to_disk('seg/%.6d_cam06seg.jpg' % img6_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img7_rgb.save_to_disk('rgb/%.6d_cam07rgb.jpg' % img7_rgb.frame)
                img7_dep.save_to_disk('dep/%.6d_cam07dep.png' % img7_dep.frame, carla.ColorConverter.Raw)
                img7_seg.save_to_disk('seg/%.6d_cam07seg.jpg' % img7_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img8_rgb.save_to_disk('rgb/%.6d_cam08rgb.jpg' % img8_rgb.frame)
                img8_dep.save_to_disk('dep/%.6d_cam08dep.png' % img8_dep.frame, carla.ColorConverter.Raw)
                img8_seg.save_to_disk('seg/%.6d_cam08seg.jpg' % img8_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img9_rgb.save_to_disk('rgb/%.6d_cam09rgb.jpg' % img9_rgb.frame)
                img9_dep.save_to_disk('dep/%.6d_cam09dep.png' % img9_dep.frame, carla.ColorConverter.Raw)
                img9_seg.save_to_disk('seg/%.6d_cam09seg.jpg' % img9_seg.frame,
                                      carla.ColorConverter.CityScapesPalette)
                img10_rgb.save_to_disk('rgb/%.6d_cam10rgb.jpg' % img10_rgb.frame)
                img10_dep.save_to_disk('dep/%.6d_cam10dep.png' % img10_dep.frame, carla.ColorConverter.Raw)
                img10_seg.save_to_disk('seg/%.6d_cam10seg.jpg' % img10_seg.frame,
                                       carla.ColorConverter.CityScapesPalette)
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
