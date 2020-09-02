from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage import io
from skimage.filters import threshold_yen
from skimage.transform import hough_line, hough_line_peaks


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
    # pad with zeros to square
    square_map = np.zeros((np.max(binary_map.shape), np.max(binary_map.shape)), dtype=bool)
    square_map[:binary_map.shape[0], :binary_map.shape[1]] = binary_map
    binary_map = square_map
    return binary_map


# house_expo_dir="/home/tzkr/python_workspace/HouseExpo/experiments/map_id_100/"
house_expo_dir = "/home/tzkr/python_workspace/map_quality/Maps"

mapfiles = [f for f in listdir(house_expo_dir) if isfile(join(house_expo_dir, f))]
# mapfiles=["orkla_map.pgm"]
mapfiles.sort()
for mf_full in mapfiles:
    image = (io.imread(join(house_expo_dir, mf_full)))
    image = load_map(image)
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(image, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(mf_full, fontsize=16)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.show()
