import argparse
import logging
import math
import pickle
import statistics as stat
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
from numpy import random
from skimage import io
from skimage.filters import threshold_yen
from skimage.transform import hough_line, hough_line_peaks
from skimage.util import img_as_ubyte
from tqdm import tqdm

import helpers as he
from extended_validator import ExtendedValidator


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


########################################################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015 Matt Nedrich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MIN_DISTANCE = 0.000001

GROUP_DISTANCE_TOLERANCE = .1


def distance_wrap_2d_vec(p1, p2):
    diff = np.subtract(p1, p2)
    r = np.hsplit(diff, 2)
    ar = r[0].flatten()
    lr = r[1].flatten()
    ad = abs(wrap_to_pi_vec(ar))
    ld = abs(lr)
    ad_ad = np.multiply(ad, ad)
    ld_ld = np.multiply(ld, ld)
    dist = np.sqrt(ad_ad + ld_ld)
    return dist


class PointGrouper(object):
    def __init__(self, distance=distance_wrap_2d_vec):
        self.distance = distance

    def group_points(self, points):
        group_assignment = []
        groups = []
        group_index = 0
        for point in points:
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index is None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
        return np.array(group_assignment)

    def _determine_nearest_group(self, point, groups):
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    def _distance_to_group(self, point, group):
        min_distance = sys.float_info.max
        for pt in group:
            dist = self.distance(point, pt)
            if dist < min_distance:
                min_distance = dist
        return min_distance


def mean_2d_vec(p):
    a = p[:, 0]
    le = p[:, 1]

    c = np.sum(np.cos(a)) / len(a)
    s = np.sum(np.sin(a)) / len(a)

    if c >= 0:
        cr_m = np.arctan(s / c)
    else:
        cr_m = np.arctan(s / c) + math.pi
    l_m = np.sum(le) / len(le)
    mean = [wrap_to_2pi(cr_m), l_m]
    return mean


def wrap_to_pi_vec(a):
    res_1 = ((a + math.pi) % (2 * math.pi) - math.pi) * ((a < -math.pi) | (a > math.pi))
    res_2 = a * ~((a < -math.pi) | (a > math.pi))
    res = res_1 + res_2
    return res


def wrap_to_2pi(a):
    if (a < 0) or (a > 2 * math.pi):
        a = abs(a % (2 * math.pi))
    else:
        a = a
    return a


def weighted_mean_2d_vec(p, w):
    a = p[:, 0]
    le = p[:, 1]

    c = np.sum(np.multiply(np.cos(a), w)) / np.sum(w)
    s = np.sum(np.multiply(np.sin(a), w)) / np.sum(w)

    if c >= 0:
        cr_m = np.arctan(s / c)
    else:
        cr_m = np.arctan(s / c) + math.pi
    l_m = np.sum(np.multiply(le, w)) / np.sum(w)
    mean = [wrap_to_2pi(cr_m), l_m]
    return mean


def gaussian_kernel(distance, bandwidth):
    # euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)
    return val


class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel, distance=distance_wrap_2d_vec, weight=weighted_mean_2d_vec):
        self.kernel = kernel
        self.distance = distance
        self.weight = weight

    def cluster(self, points, kernel_bandwidth, iteration_callback=None):
        if iteration_callback:
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        history = points
        history = history.tolist()
        for i in range(0, len(history)):
            history[i] = [history[i]]

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)

                dist = self.distance(p_new, p_new_start)

                history[i].append(p_new)
                # print(history[i])

                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        point_grouper = PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())

        return MeanShiftResult(points, shift_points, group_assignments, history)

    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        points = np.array(points)
        point_rep = np.tile(point, [len(points), 1])
        dist = self.distance(point_rep, points)
        point_weights = self.kernel(dist, kernel_bandwidth)

        shifted_point = self.weight(points, point_weights)
        return shifted_point


class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids, history):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids
        self.history = history
        self.mixing_factors = []
        self.covariances = []
        self.mean_values = []
        # compute GMM parameters
        unique_cluster_ids, counts = np.unique(self.cluster_ids, return_counts=True)
        for uid, c in zip(unique_cluster_ids, counts):
            self.mixing_factors.append(c / self.cluster_ids.size)
            self.mean_values.append(mean_2d_vec(self.original_points[self.cluster_ids == uid, :]))
            self.covariances.append(np.cov(self.original_points[self.cluster_ids == uid, :]))


########################################################################################################################

def min_wraped_distance(x1, x2):
    dx1 = abs(x1 - x2)
    dx2 = np.pi
    if (dx1 > np.pi / 2):
        dx2 = np.pi - dx1
    return min(dx1, dx2)


def hough_angles(in_map):
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    tested_angles = np.linspace(0, np.pi, 360)
    h, theta, d = hough_line(in_map, theta=tested_angles)
    _, angle, _ = hough_line_peaks(h, theta, d)
    # angle = np.array(list(angle))
    small_noise = random.rand(angle.size)
    angle = np.transpose(np.vstack((angle, small_noise)))

    mean_shifter = MeanShift()
    mean_shift_result = mean_shifter.cluster(angle, kernel_bandwidth=0.3)
    r_ang = mean_shift_result.mean_values
    # print("-------------------")
    # print(len(r_ang),r_ang)
    r_ang = [np.pi / 2 - r[0] for r in r_ang]
    # print(len(r_ang),r_ang)
    return r_ang


def pairwise_distance(pair1, pair2):
    dist = []
    for x1 in pair1:
        for x2 in pair2:
            x1 = x1 if x1 < np.pi else x1 - np.pi
            x2 = x2 if x2 < np.pi else x2 - np.pi
            dist.append(min_wraped_distance(x1, x2))
    return min(dist)


########################################################################################################################
tqdm_flag = False

now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(filename='logs/hough_' + date_time + '.log', level=logging.DEBUG,
                    format='%(levelname)s:%(module)s:%(lineno)d:%(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='JSON configuration file')
parser.add_argument('schema_file', help='JSON schema file')
args = parser.parse_args()

json_validation = ExtendedValidator(args.config_file, args.schema_file)

success, config = json_validation.extended_validator()
logging.debug("configuration: %s", config)
if not success:
    he.eprint(config)
    exit(-1)

# Logging
logging.debug("config file: %s", args.config_file)
logging.debug("schema file: %s", args.schema_file)

# search the house expo dir
house_expo_dir = config["input_directory"]
mapfiles = [f for f in listdir(house_expo_dir) if isfile(join(house_expo_dir, f))]
mapfiles.sort()
batches = []
error_names = ["obs_count", "line_err", "ang_error", "laser_noise"]
error_types = {i: set() for i in error_names}

for mf_full in mapfiles:
    mf = mf_full[:-4]
    split_mf = mf.split("_")

    map_identiffication = {'id': split_mf[0], 'file_name': mf_full, error_names[0]: float(split_mf[3]),
                           error_names[1]: float(split_mf[6]), error_names[2]: float(split_mf[9]),
                           error_names[3]: float(split_mf[12])}

    for en in error_names:
        if map_identiffication[en] != 0.0:
            error_types[en].add(map_identiffication[en])

    logging.debug("found map: %s", mf)
    if map_identiffication['id'] in [x['id'] for x in batches]:
        it = next(item for item in batches if item["id"] == map_identiffication['id'])
        it['variations'].append(map_identiffication)
        logging.debug("updating batch: %s", map_identiffication['id'])
    else:
        lb = {"id": map_identiffication['id'], 'variations': [map_identiffication]}
        batches.append(lb)
        logging.debug("new batch: %s", map_identiffication['id'])
for en in error_names:
    error_types[en] = list(error_types[en])
    error_types[en].sort()
#
# for et_key in error_types:
#     error_types[et_key] = {i: {"values": [], "stats": {}} for i in error_types[et_key]}

for et_key in error_types:
    error_types[et_key] = {
        i: {"ang_errors": [], "directions_count_error": [], "stats_ang": {}, "stats_directions_count": {}} for i in
        error_types[et_key]}

# computing angles
all_error_matrix = []
max_dir_count = 0
for batch in tqdm(batches, desc="Processed environments", disable=tqdm_flag):
    # compute orientations
    for variant in tqdm(batch['variations'], desc="finding directions " + batch['id'], disable=tqdm_flag):
        grid_map = img_as_ubyte(io.imread(join(house_expo_dir, variant['file_name'])))
        grid_map = load_map(grid_map)

        variant['directions'] = hough_angles(grid_map)
        # find reference for the variant
        if variant[error_names[0]] == 0.0 and variant[error_names[1]] == 0.0 and variant[error_names[2]] == 0.0 and \
                variant[error_names[3]] == 0.0:
            ref_map = hough_angles(grid_map)
    # compute errors
    for variant in tqdm(batch['variations'], desc="computing error " + batch['id'], disable=tqdm_flag):
        variant['errors'] = []
        variant['directions_count_error'] = []
        if not (variant[error_names[0]] == 0.0 and variant[error_names[1]] == 0.0 and variant[error_names[2]] == 0.0 and
                variant[error_names[3]] == 0.0):
            for d in variant['directions']:
                min_err = np.pi
                for r in ref_map:
                    # min_err = min(min_err, pairwise_distance(d, r))
                    min_err = min(min_err, min_wraped_distance(d, r))
                variant['errors'].append(min_err)
            variant['directions_count_error'].append(abs(len(ref_map) - len(variant['directions'])))

            for en in error_names:
                if variant[en] != 0.0:
                    # error_types[en][variant[en]]["values"].extend(variant['errors'])
                    error_types[en][variant[en]]["ang_errors"].extend(variant['errors'])
                    error_types[en][variant[en]]["directions_count_error"].extend(variant['directions_count_error'])

# compute stattistics
for ket, et in error_types.items():
    for kv, v in et.items():
        v["stats_ang"] = {"mean": stat.mean(v["ang_errors"]), "std": stat.pstdev(v["ang_errors"])}
        v["stats_directions_count"] = {"mean": stat.mean(v["directions_count_error"]),
                                       "std": stat.pstdev(v["directions_count_error"])}

# print statistics
for ket, et in error_types.items():
    print(ket)
    for kv, v in et.items():
        print("{:.2f}:  angular M={:.3f} std={:.3f}, directions M={:.3f} std={:.3f}".format(kv, v["stats_ang"]["mean"],
                                                                                            v["stats_ang"]["std"],
                                                                                            v["stats_directions_count"][
                                                                                                "mean"],
                                                                                            v["stats_directions_count"][
                                                                                                "std"]))
        logging.debug("%s, %.2f:  angular M=%.3f std=%.3f, directions M=%.3f std=%.3f", ket, kv, v["stats_ang"]["mean"],
                      v["stats_ang"]["std"],
                      v["stats_directions_count"][
                          "mean"],
                      v["stats_directions_count"][
                          "std"])
        # logging.debug("%s, %.2f: M=%.3f std=%.3f" % (ket, kv, v["stats_ang"]["mean"], v["stats_ang"]["std"]))

f = open("results/dir_error_types_Hough_90.pkl", "wb")
pickle.dump(error_types, f)
f.close()

f = open("results/dir_batches_Hough_90.pkl", "wb")
pickle.dump(batches, f)
f.close()
