# generate 10 cameras of panoptic segmention img and its gt.
import carla
import pygame
import queue
import numpy as np
import cv2

def set_cam(world, img_width, img_height, fov):
    # Spawn attached RGB camera
    cam_rgb = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_rgb.set_attribute("image_size_x", str(img_width))
    cam_rgb.set_attribute("image_size_y", str(img_height))
    cam_rgb.set_attribute("fov", str(fov))

    insseg = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
    insseg.set_attribute('image_size_x', str(img_width))
    insseg.set_attribute('image_size_y', str(img_height))
    insseg.set_attribute('fov', str(fov))
    return [cam_rgb, insseg]


class CarlaSyncMode(object):
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


def main():
    actor_list = []
    pygame.init()

    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    try:

        [cam_rgb, insseg] = set_cam(world, img_width=2048, img_height=1024, fov=120)

        cam1_transform = carla.Transform(carla.Location(x=-35.0, y=28.8, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-180.0, roll=0.000000))
        cam1_rgb = world.spawn_actor(cam_rgb, cam1_transform)
        actor_list.append(cam1_rgb)
        cam1_ins = world.spawn_actor(insseg, cam1_transform)
        actor_list.append(cam1_ins)

        cam2_transform = carla.Transform(carla.Location(x=-30.0, y=-66.0, z=6.3),
                                         carla.Rotation(pitch=-35.000, yaw=135.0, roll=0.000000))
        cam2_rgb = world.spawn_actor(cam_rgb, cam2_transform)
        actor_list.append(cam2_rgb)
        cam2_ins = world.spawn_actor(insseg, cam2_transform)
        actor_list.append(cam2_ins)

        cam3_transform = carla.Transform(carla.Location(x=-106.0, y=25.0, z=6.2),
                                         carla.Rotation(pitch=-37.000, yaw=50.0, roll=0.000000))
        cam3_rgb = world.spawn_actor(cam_rgb, cam3_transform)
        actor_list.append(cam3_rgb)
        cam3_ins = world.spawn_actor(insseg, cam3_transform)
        actor_list.append(cam3_ins)

        cam4_transform = carla.Transform(carla.Location(x=-50.0, y=3.0, z=6.1),
                                         carla.Rotation(pitch=-38.000, yaw=-90.0, roll=0.000000))
        cam4_rgb = world.spawn_actor(cam_rgb, cam4_transform)
        actor_list.append(cam4_rgb)
        cam4_ins = world.spawn_actor(insseg, cam4_transform)
        actor_list.append(cam4_ins)

        cam5_transform = carla.Transform(carla.Location(x=106.0, y=41.0, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=-125.0, roll=0.000000))
        cam5_rgb = world.spawn_actor(cam_rgb, cam5_transform)
        actor_list.append(cam5_rgb)
        cam5_ins = world.spawn_actor(insseg, cam5_transform)
        actor_list.append(cam5_ins)

        cam6_transform = carla.Transform(carla.Location(x=47.0, y=70.0, z=6.4),
                                         carla.Rotation(pitch=-39.000, yaw=0.0, roll=0.000000))
        cam6_rgb = world.spawn_actor(cam_rgb, cam6_transform)
        actor_list.append(cam6_rgb)
        cam6_ins = world.spawn_actor(insseg, cam6_transform)
        actor_list.append(cam6_ins)

        cam7_transform = carla.Transform(carla.Location(x=103, y=100.0, z=6.2),
                                         carla.Rotation(pitch=-37.000, yaw=-135.0, roll=0.000000))
        cam7_rgb = world.spawn_actor(cam_rgb, cam7_transform)
        actor_list.append(cam7_rgb)
        cam7_ins = world.spawn_actor(insseg, cam7_transform)
        actor_list.append(cam7_ins)

        cam8_transform = carla.Transform(carla.Location(x=30, y=69.0, z=6.2),
                                         carla.Rotation(pitch=-38.000, yaw=-40.0, roll=0.000000))
        cam8_rgb = world.spawn_actor(cam_rgb, cam8_transform)
        actor_list.append(cam8_rgb)
        cam8_ins = world.spawn_actor(insseg, cam8_transform)
        actor_list.append(cam8_ins)

        cam9_transform = carla.Transform(carla.Location(x=-113, y=-48, z=6.5),
                                         carla.Rotation(pitch=-40.000, yaw=50.0, roll=0.000000))
        cam9_rgb = world.spawn_actor(cam_rgb, cam9_transform)
        actor_list.append(cam9_rgb)
        cam9_ins = world.spawn_actor(insseg, cam9_transform)
        actor_list.append(cam9_ins)

        cam10_transform = carla.Transform(carla.Location(x=-47.0, y=5.0, z=6.5),
                                          carla.Rotation(pitch=-40.000, yaw=160.0, roll=0.000000))
        cam10_rgb = world.spawn_actor(cam_rgb, cam10_transform)
        actor_list.append(cam10_rgb)
        cam10_ins = world.spawn_actor(insseg, cam10_transform)
        actor_list.append(cam10_ins)
        count = 0
        # Create a synchronous mode context.
        with CarlaSyncMode(world, cam1_rgb, cam1_ins, cam2_rgb, cam2_ins, cam3_rgb, cam3_ins,cam4_rgb, cam4_ins,cam5_rgb, cam5_ins,cam6_rgb, cam6_ins,cam7_rgb, cam7_ins,cam8_rgb, cam8_ins,cam9_rgb, cam9_ins,cam10_rgb, cam10_ins,fps=40) as sync_mode:
            for frame in range(0, 3000):
            # while True:
                clock.tick()
                # Advance the simulation and wait for the data.
                snapshot, img1_rgb, img1_ins, img2_rgb, img2_ins, img3_rgb, img3_ins, img4_rgb, img4_ins, img5_rgb, img5_ins, img6_rgb, img6_ins, img7_rgb, img7_ins, img8_rgb, img8_ins, img9_rgb, img9_ins, img10_rgb, img10_ins  = sync_mode.tick(timeout=2.0)
                if frame % 5 == 0:
                    img1_rgb.save_to_disk('rgb/%.6d_cam01rgb.png' % img1_rgb.frame)
                    img2_rgb.save_to_disk('rgb/%.6d_cam02rgb.png' % img2_rgb.frame)
                    img3_rgb.save_to_disk('rgb/%.6d_cam03rgb.png' % img3_rgb.frame)
                    img4_rgb.save_to_disk('rgb/%.6d_cam04rgb.png' % img4_rgb.frame)
                    img5_rgb.save_to_disk('rgb/%.6d_cam05rgb.png' % img5_rgb.frame)
                    img6_rgb.save_to_disk('rgb/%.6d_cam06rgb.png' % img6_rgb.frame)
                    img7_rgb.save_to_disk('rgb/%.6d_cam07rgb.png' % img7_rgb.frame)
                    img8_rgb.save_to_disk('rgb/%.6d_cam08rgb.png' % img8_rgb.frame)
                    img9_rgb.save_to_disk('rgb/%.6d_cam09rgb.png' % img9_rgb.frame)
                    img10_rgb.save_to_disk('rgb/%.6d_cam10rgb.png' % img10_rgb.frame)
                    img1_ins.save_to_disk('ins/%.6d_cam01ins.png' % img1_ins.frame)
                    img2_ins.save_to_disk('ins/%.6d_cam02ins.png' % img2_ins.frame)
                    img3_ins.save_to_disk('ins/%.6d_cam03ins.png' % img3_ins.frame)
                    img4_ins.save_to_disk('ins/%.6d_cam04ins.png' % img4_ins.frame)
                    img5_ins.save_to_disk('ins/%.6d_cam05ins.png' % img5_ins.frame)
                    img6_ins.save_to_disk('ins/%.6d_cam06ins.png' % img6_ins.frame)
                    img7_ins.save_to_disk('ins/%.6d_cam07ins.png' % img7_ins.frame)
                    img8_ins.save_to_disk('ins/%.6d_cam08ins.png' % img8_ins.frame)
                    img9_ins.save_to_disk('ins/%.6d_cam09ins.png' % img9_ins.frame)
                    img10_ins.save_to_disk('ins/%.6d_cam10ins.png' % img10_ins.frame)
                    print(count)
                    count+=1
                    # img = np.reshape(np.copy(img3_rgb.raw_data), (1024, 2048, 4))
                    # cv2.imshow('camera_location_test',img)
                    # cv2.waitKey(0)
                    # img_ins = np.reshape(np.copy(img1_ins.raw_data), (1024, 2048, 4)) 
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
