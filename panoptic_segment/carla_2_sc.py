# change carla format panoptic img into cityscapes format (the same as *_instanceIds.png in origen cityscapes datasets gtFine forder).
from unicodedata import category
import cv2
import numpy as np
from collections import Counter
import os
import tqdm

# for _visualize
import random
import colorsys


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def _visualize(img, id, id_image, category):
    # generate n colors for visualize
    n_colors = ncolors(len(id))

    image_imageneat = np.zeros(img.shape, dtype=np.uint16)
    for i in range(0, len(id)):
        # get instance pix
        car_pix = (id_image == id[i])
        image_imageneat[car_pix] = n_colors[i]
    image_imageneat = image_imageneat.astype(np.uint8)
    cv2.imshow('img', image_imageneat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if category == 'person':
        print("%d persons in current image." % len(id))
    elif category == 'car':
        print("%d cars in current image." % len(id))
    else:
        AssertionError


def process_one_frame(ins_source_name, ins_dst_name):
    img = cv2.imread(ins_source_name)
    img_R_channal = img[:, :, 2]
    img_G_channal = img[:, :, 1]
    img_B_channal = img[:, :, 0]

    # process cars
    cars_pix = img_R_channal == 10

    # encode car_id in to cars_id_img.(the same size as img)
    cars_id_img = img_B_channal * 1000 + img_G_channal
    cars_id_img[cars_pix == False] = 0

    # get car id
    # remove cars below 50 pix
    car_id_pix_count = list(Counter(cars_id_img.flatten()).values())
    car_id_with_noise = list(Counter(cars_id_img.flatten()).keys())

    car_id = []
    for i in range(0, len(car_id_pix_count)):
        if car_id_pix_count[i] > 100:
            car_id.append(car_id_with_noise[i])
    car_id.remove(0)

    # # visualize the car instance image
    # _visualize(img, car_id, cars_id_img, category= 'car')

    # process persons
    persons_pix = img_R_channal == 4

    # encode person_id in to persons_id_img.(the same size as img)
    persons_id_img = img_B_channal * 1000 + img_G_channal
    persons_id_img[persons_pix == False] = 0

    # get person id
    # remove persons below 50 pix
    person_id_pix_count = list(Counter(persons_id_img.flatten()).values())
    person_id_with_noise = list(Counter(persons_id_img.flatten()).keys())

    person_id = []
    for i in range(0, len(person_id_pix_count)):
        if person_id_pix_count[i] > 100:
            person_id.append(person_id_with_noise[i])
    person_id.remove(0)

    # # visualize the person instance image
    # _visualize(img, person_id, persons_id_img, category= 'person')

    # visualize the car & person instance image
    cityscape_format_image = np.zeros(
        (img.shape[0], img.shape[1]), dtype=np.uint16)
    for i in range(0, len(car_id)):
        # get car instance pix
        car_pix = (cars_id_img == car_id[i])
        cityscape_format_image[car_pix] = 26001+i
    for i in range(0, len(person_id)):
        # get person instance pix
        car_pix = (persons_id_img == person_id[i])
        cityscape_format_image[car_pix] = 24001+i
    # test = np.any(persons_image == 26000)

    cv2.imwrite(ins_dst_name, cityscape_format_image)


if __name__ == "__main__":

    ins_path = "/media/whn/新加卷/dataset/carla_ins/panoptic_dataset_cityscape_format/ins"
    ins_dst = "/media/whn/新加卷/dataset/carla_ins/panoptic_dataset_cityscape_format/ins_sc_format"

    ins_files = os.listdir(ins_path)
    for ins_name in tqdm.tqdm(ins_files):
        ins_source_name = os.path.join(ins_path, ins_name)
        ins_dst_name = os.path.join(ins_dst, ins_name)
        process_one_frame(ins_source_name, ins_dst_name)
