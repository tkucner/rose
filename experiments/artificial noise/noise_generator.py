import random
from enum import Enum
from os import listdir
from os.path import isfile, join

import numpy as np
import png
from skimage import io
from skimage.filters import threshold_yen
from skimage.morphology import (square, rectangle, diamond, disk, octagon, star)
from skimage.util import img_as_ubyte
from tqdm import tqdm


def load_map(grid_map):
    if len(grid_map.shape) == 3:
        grid_map = grid_map[:, :, 1]
    thresh = threshold_yen(grid_map)
    binary_map = grid_map <= thresh
    binary_map = binary_map
    if binary_map.shape[0] % 2 != 0:
        t = np.zeros((binary_map.shape[0] + 1, binary_map.shape[1]), dtype=bool)
        t[:-1, :] = binary_map
        binary_map = t
    if binary_map.shape[1] % 2 != 0:
        t = np.zeros((binary_map.shape[0], binary_map.shape[1] + 1), dtype=bool)
        t[:, :-1] = binary_map
        binary_map = t
    return binary_map


class obstacle_shape(Enum):
    SQUARE = 1
    RECTANGLE = 2
    DIAMOND = 3
    DISK = 4
    OCTAGON = 5
    STAR = 6
    RANDOM = 7


def obstacle_generator(size_limits, noise_type):
    size = (random.randint(1, size_limits), random.randint(1, size_limits))
    if noise_type == obstacle_shape.SQUARE:
        noise = square(size[1])
    if noise_type == obstacle_shape.RECTANGLE:
        noise = rectangle(size[0], size[1])
    if noise_type == obstacle_shape.DIAMOND:
        noise = diamond(size[0])
    if noise_type == obstacle_shape.DISK:
        noise = disk(size[0])
    if noise_type == obstacle_shape.OCTAGON:
        noise = octagon(size[0], size[1])
    if noise_type == obstacle_shape.STAR:
        noise = star(size[0])
    return noise


def generate_noise_single_type(mask_size, cout, size_limits, noise_type):
    mask = np.zeros(mask_size)
    for _ in range(cout):
        if noise_type == obstacle_shape.RANDOM:
            noise_type_u = random.choice(list(obstacle_shape)[:-1])
        else:
            noise_type_u = noise_type
        noise = obstacle_generator(size_limits, noise_type_u)
        x = random.randint(1, mask.shape[0] - noise.shape[0] - 1)
        y = random.randint(1, mask.shape[1] - noise.shape[1] - 1)
        temp_mask = np.zeros(mask_size)
        temp_mask[x:x + noise.shape[0], y:y + noise.shape[1]] = noise
        mask = np.logical_or(mask, temp_mask)
    return mask


def add_noise(binar_map, mask):
    return np.logical_or(mask, binar_map)


map_dir = "/home/tzkr/python_workspace/rose/experiments/general_map_evaluation/Cirill_maps"
mapfiles = [f for f in listdir(map_dir) if isfile(join(map_dir, f))]
mapfiles.sort()

save_dir = "/home/tzkr/python_workspace/rose/experiments/artificial noise/maps"
for m in tqdm(mapfiles, desc="environment"):
    if m.find('bad') == -1:
        name = m.split('.')[0]
        grid_map = img_as_ubyte(io.imread(join(map_dir, m)))
        grid_map = load_map(grid_map)
        for count in tqdm(range(20, 200, 20), desc="obstacle count"):
            for size in tqdm(range(2, 20), desc="obstacle size"):
                for ty in [obstacle_shape.SQUARE, obstacle_shape.RECTANGLE, obstacle_shape.RANDOM]:
                    mask = generate_noise_single_type(grid_map.shape, count, size, ty)
                    noisy_map = add_noise(grid_map, mask)
                    error_type = (str(ty).split('.')[1])
                    save_name = name + "_ocount_" + str(count) + "_osize_" + str(size) + "_otype_" + error_type + ".png"
                    noisy_map = noisy_map * 255
                    noisy_map = noisy_map.astype(np.uint8)

                    png.from_array(np.array(noisy_map), 'L').save(join(save_dir, save_name))
