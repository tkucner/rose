from os import listdir
from os.path import join

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.util import img_as_ubyte

reference_dir = "ref_maps"
reference_map_files = listdir(reference_dir)
reference_maps = {}

test_dir = "test_maps/NOFFT"
test_map_files = listdir(reference_dir)
test_maps = {}


# load test map

def find_lines_pxl(map):
    # find colors
    m = list(map.reshape(-1, 4))
    res_lin = np.array([0 if x[0] > 128 and x[1] < 128 and x[2] < 128 else 1 for x in m])
    res = res_lin.reshape(map.shape[0:2])
    return res


def score_maps(ref, test):
    # simple ide compute the mean distance to closest pixel
    ref_dist = ndimage.distance_transform_edt(ref)
    negative_ref = np.negative(ref) + np.ones(ref.shape)
    negative_test = np.negative(test) + np.ones(test.shape)

    dist_score_inp = np.sum(np.sum(np.multiply(ref_dist, negative_test)))
    dist_denom = np.sum(np.sum(negative_test))

    dist_score = dist_score_inp / dist_denom

    count_score = np.sum(np.sum(negative_test)) / np.sum(np.sum(negative_ref))
    # fig, axes = plt.subplots(1, 3)
    # ax = axes.ravel()
    #
    # ax[0].imshow(ref, cmap=cm.gray)
    # ax[0].set_axis_off()
    #
    # ax[1].imshow(ref_dist, cmap=cm.gray)
    # ax[1].set_axis_off()
    #
    # ax[2].imshow(negative_test, cmap=cm.gray)
    # ax[2].set_axis_off()
    #
    # plt.tight_layout()
    # plt.show()

    return dist_score, count_score


for rmf in reference_map_files:
    ref_map = img_as_ubyte(io.imread(join(reference_dir, rmf)))
    reference_maps[rmf.split(".")[0]] = {"map": ref_map, "line_map": find_lines_pxl(ref_map)}

for tmf in test_map_files:
    test_map = img_as_ubyte(io.imread(join(test_dir, tmf)))
    test_maps[tmf.split(".")[0]] = {"map": test_map, "line_map": find_lines_pxl(test_map)}

for tm_key in test_maps:
    ref = reference_maps[tm_key]["line_map"]
    test = test_maps[tm_key]["line_map"]
    print(score_maps(ref, test))
