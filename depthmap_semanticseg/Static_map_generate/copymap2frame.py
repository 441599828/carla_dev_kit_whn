# copy map to each frame
import os
from shutil import copyfile
from tqdm import tqdm
from PIL import Image, ImageChops


def front_mask(frame, background):
    frameimg = Image.open(frame)
    backgroundimg = Image.open(background)

    # dif between map and frame. (to avoid car-in-map effect)
    diff = ImageChops.difference(frameimg, backgroundimg).convert('L')
    threshold = 30
    table1 = []
    for i in range(256):
        if i < threshold:
            table1.append(0)
        else:
            table1.append(1)
    mask1 = diff.point(table1, '1')

    # car and ped in frame. (to avoid tree shake effect)
    table2 = []
    for i in range(256):
        # if i > 15 and i <30:
        if (83 < i < 86) or (14 < i < 31):
            table2.append(1)
        else:
            table2.append(0)

    frameimggray = frameimg.convert('L')
    mask2 = frameimggray.point(table2, '1')

    mask = ImageChops.logical_and(mask1, mask2)
    mask.save(frame.replace('/seg/', '/frontmask/').replace('seg.jpg', 'mask.png'))


if __name__ == "__main__":
    map = "../../Data/static_map"

    # train_rgb_folder = "../../Data/train/rgb"
    # val_rgb_folder = "../../Data/val/rgb"
    # test_easy_rgb_folder = "../../Data/test_easy/rgb"
    # test_middle_rgb_folder = "../../Data/test_middle/rgb"
    # test_hard_rgb_folder = "../../Data/test_hard/rgb"
    test_hardest_rgb_folder = "../../Data/test_hardest/rgb"

    os.makedirs(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'))
    os.makedirs(test_hardest_rgb_folder.replace('/rgb', '/frontmask'))
    # os.makedirs(test_hardest_rgb_folder.replace('/rgb', '/depmap'))

    os.makedirs(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'))
    os.makedirs(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'))

    for img_name in tqdm(os.listdir(test_hardest_rgb_folder)):
        if 'cam01rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam01rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam01rgb.jpg',
                                                   'cam01rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam01dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam01rgb.jpg',
            #                                        'cam01depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam01seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar1.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam01rgb.jpg',
                                                                                                      'lidar01.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar1.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam01rgb.jpg',
                                                                                                'lidar01.pcd')))

        elif 'cam02rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam02rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam02rgb.jpg',
                                                   'cam02rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam02dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam02rgb.jpg',
            #                                        'cam02depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam02seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar2.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam02rgb.jpg',
                                                                                                      'lidar02.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar2.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam02rgb.jpg',
                                                                                                'lidar02.pcd')))
        elif 'cam03rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam03rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam03rgb.jpg',
                                                   'cam03rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam03dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam03rgb.jpg',
            #                                        'cam03depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam03seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar3.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam03rgb.jpg',
                                                                                                      'lidar03.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar3.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam03rgb.jpg',
                                                                                                'lidar03.pcd')))
        elif 'cam04rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam04rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam04rgb.jpg',
                                                   'cam04rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam04dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam04rgb.jpg',
            #                                        'cam04depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam04seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar4.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam04rgb.jpg',
                                                                                                      'lidar04.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar4.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam04rgb.jpg',
                                                                                                'lidar04.pcd')))
        elif 'cam05rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam05rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam05rgb.jpg',
                                                   'cam05rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam05dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam05rgb.jpg',
            #                                        'cam05depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam05seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar5.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam05rgb.jpg',
                                                                                                      'lidar05.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar5.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam05rgb.jpg',
                                                                                                'lidar05.pcd')))
        elif 'cam06rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam06rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam06rgb.jpg',
                                                   'cam06rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam06dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam06rgb.jpg',
            #                                        'cam06depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam06seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar6.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam06rgb.jpg',
                                                                                                      'lidar06.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar6.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam06rgb.jpg',
                                                                                                'lidar06.pcd')))
        elif 'cam07rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam07rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam07rgb.jpg',
                                                   'cam07rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam07dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam07rgb.jpg',
            #                                        'cam07depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam07seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar7.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam07rgb.jpg',
                                                                                                      'lidar07.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar7.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam07rgb.jpg',
                                                                                                'lidar07.pcd')))
        elif 'cam08rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam08rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam08rgb.jpg',
                                                   'cam08rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam08dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam08rgb.jpg',
            #                                        'cam08depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam08seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar8.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam08rgb.jpg',
                                                                                                      'lidar08.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar8.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam08rgb.jpg',
                                                                                                'lidar08.pcd')))
        elif 'cam09rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam09rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam09rgb.jpg',
                                                   'cam09rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam09dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam09rgb.jpg',
            #                                        'cam09depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam09seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar9.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam09rgb.jpg',
                                                                                                      'lidar09.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar9.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam09rgb.jpg',
                                                                                                'lidar09.pcd')))
        elif 'cam10rgb' in img_name:
            copyfile(os.path.join(map, 'from_camera/rgb/cam10rgb.jpg'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/rgbmap'),
                                  img_name.replace('cam10rgb.jpg',
                                                   'cam10rgbmap.jpg')))
            # copyfile(os.path.join(map, 'from_camera/depth/cam10dep.png'),
            #          os.path.join(test_hardest_rgb_folder.replace('/rgb', '/depmap'),
            #                       img_name.replace('cam10rgb.jpg',
            #                                        'cam10depmap.png')))
            front_mask(os.path.join(test_hardest_rgb_folder.replace('/rgb', '/seg'), img_name.replace('rgb', 'seg')),
                       os.path.join(map, 'from_camera/seg/cam10seg.jpg'))
            copyfile(os.path.join(map, 'from_lidar/depth/lidar10.png'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/sparsedepmap'), img_name.replace('cam10rgb.jpg',
                                                                                                      'lidar10.png')))
            copyfile(os.path.join(map, 'from_lidar/pcd/lidar10.pcd'),
                     os.path.join(test_hardest_rgb_folder.replace('/rgb', '/pcdmap'), img_name.replace('cam10rgb.jpg',
                                                                                                'lidar10.pcd')))
        else:
            assert ('unknown error!')
