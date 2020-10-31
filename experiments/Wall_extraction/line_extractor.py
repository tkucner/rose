from os import listdir
from os.path import join, isfile

import numpy as np
from PIL import Image
from shapely.geometry import LineString
from skimage import transform
from skimage.filters import threshold_mean
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks


def detect_lines(image):
    skelton_image = skeletonize(image)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1440)
    h, theta, d = hough_line(skelton_image, theta=tested_angles)

    e = [LineString([(0, skelton_image.shape[0]), (0, 0)]),
         LineString([(0, 0), (skelton_image.shape[1], 0)]),
         LineString([(skelton_image.shape[1], 0), (skelton_image.shape[1], skelton_image.shape[0])]),
         LineString([(skelton_image.shape[1], skelton_image.shape[0]), (0, skelton_image.shape[0])])]

    origin = np.array((-1.0, skelton_image.shape[1] + 1))
    intervals = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        l = LineString([(origin[0], y0), (origin[1], y1)])
        interval = []
        for ei in e:
            intersection = l.intersection(ei)
            if not intersection.is_empty:
                interval.append([intersection.x, intersection.y])
        intervals.append(np.array(interval))
    return intervals


def binarize_image(inp_map):
    img = Image.open(inp_map).convert('L')
    thresh = threshold_mean(img)
    binary = img < thresh
    binary = binary * 1
    return binary


def compute_transform(l1, l2):
    tform = transform.estimate_transform('euclidean', l1, l2)
    # tform = transform.estimate_transform('affine', l1, l2)
    return tform


def compute_score(l1, l2):
    tform = compute_transform(l1, l2)
    r = np.sqrt(tform.translation[0] ** 2 + tform.translation[1] ** 2)
    score = np.pi * r * np.abs(tform.rotation) / (2 * np.pi)
    return score


# map directory
map_dir = "Matteo_Maps"
# list all maps
map_files = [f for f in listdir(map_dir) if isfile(join(map_dir, f))]
map_names = list(set([(f.split(".")[0]).rsplit("_", 1)[0] for f in map_files]))
# split list according to map type
reference_map_files = list(filter(None, [f if "GT" in f else None for f in map_files]))
Hough_map_files = list(filter(None, [f if "HG" in f else None for f in map_files]))
FFT_map_files = list(filter(None, [f if "FFT" in f else None for f in map_files]))

tuples = []
for map_name in map_names:
    tuples.append({"name": map_name,
                   "ref": [i for i in reference_map_files if map_name in i],
                   "HG": [i for i in Hough_map_files if map_name in i],
                   "FFT": [i for i in FFT_map_files if map_name in i]})

id = 0
labels = []
for t in tuples:
    if [] in t.values():
        pass
    else:
        # print("Processing: {}".format(t["name"]))
        labels.append(t["name"])
        t["ref_map"] = ref_map = binarize_image(join(map_dir, t["ref"][0]))
        t["HG_map"] = HG_map = binarize_image(join(map_dir, t["HG"][0]))
        t["FFT_map"] = FFT_map = binarize_image(join(map_dir, t["FFT"][0]))

        t["ref_lines"] = list(filter(lambda x: x.size != 0, detect_lines(ref_map)))
        t["HG_lines"] = list(filter(lambda x: x.size != 0, detect_lines(HG_map)))
        t["FFT_lines"] = list(filter(lambda x: x.size != 0, detect_lines(FFT_map)))

        # compute scores for HG
        HG_scores = []
        for lhg in t["HG_lines"]:
            l_HG_scores = []
            for ref in t["ref_lines"]:
                l_HG_scores.append(compute_score(lhg, ref))
            HG_scores.append(min(l_HG_scores))
        # print("Hough Score: {:.3f}".format(sum(HG_scores)/len(HG_scores)))
        print("{:d} {:.3f} r".format(id, sum(HG_scores) / len(HG_scores)))
        # compute scores for HG
        FFT_scores = []
        for lfft in t["FFT_lines"]:
            l_FFT_scores = []
            for ref in t["ref_lines"]:
                l_FFT_scores.append(compute_score(lfft, ref))
            FFT_scores.append(min(l_FFT_scores))
        # print("FFT Score: {:.3f}".format(sum(FFT_scores)/len(FFT_scores)))
        print("{:d} {:.3f} g".format(id, sum(FFT_scores) / len(FFT_scores)))
        id += 1
        # for hl in HG_lines:
        #     for rl in ref_lines:
        #         tform = transform.estimate_transform('euclidean', hl, rl)
        #         print(tform.translation, tform.rotation)

print(labels)
