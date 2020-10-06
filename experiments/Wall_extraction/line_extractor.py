from os import listdir
from os.path import join, isfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_mean
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft


def binarize_image(inp_map):
    img = Image.open(inp_map).convert('L')
    thresh = threshold_mean(img)
    binary = img < thresh
    binary = binary * 1
    return binary


# map directory
map_dir = "Matteo_Maps"
# list all maps
map_files = [f for f in listdir(map_dir) if isfile(join(map_dir, f))]
map_names = list(set([(f.split(".")[0]).rsplit("_", 1)[0] for f in map_files]))
# split list according to map type
reference_map_files = list(filter(None, [f if "GT" in f else None for f in map_files]))
Hough_map_files = list(filter(None, [f if "HG" in f else None for f in map_files]))
FFT_map_files = list(filter(None, [f if "FFT" in f else None for f in map_files]))


def remove_padding(inp_map):
    col_sum = inp_map.sum(axis=0)
    inp_map = np.delete(inp_map, np.where(col_sum == 0), axis=1)

    row_sum = inp_map.sum(axis=1)
    inp_map = np.delete(inp_map, np.where(row_sum == 0), axis=0)
    return inp_map


def registration(ref_map, move_map):
    shift, error, diffphase = phase_cross_correlation(ref_map, move_map)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(ref_map, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference ref_map')

    ax2.imshow(move_map.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset ref_map')

    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    map_product = np.fft.fft2(ref_map) * np.fft.fft2(move_map).conj()
    cc_map = np.fft.fftshift(np.fft.ifft2(map_product))
    ax3.imshow(cc_map.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    plt.show()

    print(f"Detected pixel offset (y, x): {shift}")

    # subpixel precision
    shift, error, diffphase = phase_cross_correlation(ref_map, move_map,
                                                      upsample_factor=100)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(ref_map, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference ref_map')

    ax2.imshow(move_map.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    cc_image = _upsampled_dft(map_product, 150, 100, (shift * 100) + 75).conj()
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")

    plt.show()


tuples = []
for map_name in map_names:
    tuples.append({"name": map_name,
                   "ref": [i for i in reference_map_files if map_name in i],
                   "HG": [i for i in Hough_map_files if map_name in i],
                   "FFT": [i for i in FFT_map_files if map_name in i]})

for t in tuples:
    if [] in t.values():
        print("Skipping: {}".format(t["name"]))
    else:
        ref_map = binarize_image(join(map_dir, t["ref"][0]))
        HG_map = binarize_image(join(map_dir, t["HG"][0]))
        FFT_map = binarize_image(join(map_dir, t["FFT"][0]))
        remove_padding(ref_map)
        remove_padding(HG_map)
        remove_padding(FFT_map)

        fig, axes = plt.subplots(1, 3)
        ax = axes.ravel()

        ref_map = remove_padding(ref_map)
        ax[0].imshow(ref_map)
        ax[0].set_axis_off()
        print(ref_map.shape)

        HG_map = remove_padding(HG_map)
        ax[1].imshow(HG_map)
        ax[1].set_axis_off()
        print(HG_map.shape)

        FFT_map = remove_padding(FFT_map)
        ax[2].imshow(FFT_map)
        ax[2].set_axis_off()
        print(FFT_map.shape)

        plt.tight_layout()
        plt.show()

        registration(ref_map, FFT_map)

        break


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

#####################################################
# OLD CODE
#####################################################
# reference_dir = "ref_maps"
# reference_map_files = listdir(reference_dir)
# reference_maps = {}
#
# test_dir = "test_maps/NOFFT"
# test_map_files = listdir(reference_dir)
# test_maps = {}
#
#
# # load test map
#
# def find_lines_pxl(map):
#     # find colors
#     m = list(map.reshape(-1, 4))
#     res_lin = np.array([0 if x[0] > 128 and x[1] < 128 and x[2] < 128 else 1 for x in m])
#     res = res_lin.reshape(map.shape[0:2])
#     return res
#
#
# def score_maps(ref, test):
#     # simple ide compute the mean distance to closest pixel
#     ref_dist = ndimage.distance_transform_edt(ref)
#     negative_ref = np.negative(ref) + np.ones(ref.shape)
#     negative_test = np.negative(test) + np.ones(test.shape)
#
#     dist_score_inp = np.sum(np.sum(np.multiply(ref_dist, negative_test)))
#     dist_denom = np.sum(np.sum(negative_test))
#
#     dist_score = dist_score_inp / dist_denom
#
#     count_score = np.sum(np.sum(negative_test)) / np.sum(np.sum(negative_ref))
#     # fig, axes = plt.subplots(1, 3)
#     # ax = axes.ravel()
#     #
#     # ax[0].imshow(ref, cmap=cm.gray)
#     # ax[0].set_axis_off()
#     #
#     # ax[1].imshow(ref_dist, cmap=cm.gray)
#     # ax[1].set_axis_off()
#     #
#     # ax[2].imshow(negative_test, cmap=cm.gray)
#     # ax[2].set_axis_off()
#     #
#     # plt.tight_layout()
#     # plt.show()
#
#     return dist_score, count_score
#
#
# for rmf in reference_map_files:
#     ref_map = img_as_ubyte(io.imread(join(reference_dir, rmf)))
#     reference_maps[rmf.split(".")[0]] = {"map": ref_map, "line_map": find_lines_pxl(ref_map)}
#
# for tmf in test_map_files:
#     test_map = img_as_ubyte(io.imread(join(test_dir, tmf)))
#     test_maps[tmf.split(".")[0]] = {"map": test_map, "line_map": find_lines_pxl(test_map)}
#
# for tm_key in test_maps:
#     ref = reference_maps[tm_key]["line_map"]
#     test = test_maps[tm_key]["line_map"]
#     print(score_maps(ref, test))
