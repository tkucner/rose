import argparse
import logging
import statistics as stat
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from tqdm import tqdm

import helpers as he
from extended_validator import ExtendedValidator
from fft_structure_extraction import FFTStructureExtraction as structure_extraction


# Logging
# time

def pairwise_distance(pair1, pair2):
    dist = []
    for x1 in pair1:
        for x2 in pair2:
            x1 = x1 if x1 < np.pi else x1 - np.pi
            x2 = x2 if x2 < np.pi else x2 - np.pi
            dist.append(min_wraped_distance(x1, x2))
    return min(dist)


def min_wraped_distance(x1, x2):
    dx1 = abs(x1 - x2)
    dx2 = np.pi
    if (dx1 > np.pi / 2):
        dx2 = np.pi - dx1
    return min(dx1, dx2)


now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(filename='logs/rose_' + date_time + '.log', level=logging.DEBUG,
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

    map_identiffication = {'id': split_mf[0], 'file_name': mf_full}
    map_identiffication[error_names[0]] = float(split_mf[3])
    map_identiffication[error_names[1]] = float(split_mf[6])
    map_identiffication[error_names[2]] = float(split_mf[9])
    map_identiffication[error_names[3]] = float(split_mf[12])

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

for et_key in error_types:
    error_types[et_key] = {i: {"values": [], "stats": {}} for i in error_types[et_key]}

# computing angles
all_error_matrix = []
max_dir_count = 0
for batch in tqdm(batches, desc="Processed environments"):
    # compute orientations
    for variant in tqdm(batch['variations'], desc="finding directions " + batch['id']):
        grid_map = img_as_ubyte(io.imread(join(house_expo_dir, variant['file_name'])))
        rose = structure_extraction(grid_map, peak_height=config["peak_extraction_parameters"]["peak_height"],
                                    smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                                    sigma=config["peak_extraction_parameters"]["sigma"])
        rose.process_map()
        variant['directions'] = rose.dom_dirs
        # find reference for the variant
        if variant[error_names[0]] == 0.0 and variant[error_names[1]] == 0.0 and variant[error_names[2]] == 0.0 and \
                variant[error_names[3]] == 0.0:
            ref_map = rose.dom_dirs
    # compute errors
    for variant in tqdm(batch['variations'], desc="computing error " + batch['id']):
        variant['errors'] = []
        if not (variant[error_names[0]] == 0.0 and variant[error_names[1]] == 0.0 and variant[error_names[2]] == 0.0 and
                variant[error_names[3]] == 0.0):
            for d in variant['directions']:
                min_err = np.pi
                for r in ref_map:
                    min_err = min(min_err, pairwise_distance(d, r))
                variant['errors'].append(min_err)

            for en in error_names:
                if variant[en] != 0.0:
                    error_types[en][variant[en]]["values"].extend(variant['errors'])

# compute stattistics
for ket, et in error_types.items():
    for kv, v in et.items():
        v["stats"] = {"mean": stat.mean(v["values"]), "std": stat.pstdev(v["values"])}

# print statistics
for ket, et in error_types.items():
    print(ket)
    for kv, v in et.items():
        print("{:.2f}: M={:.3f} std={:.3f}".format(kv, v["stats"]["mean"], v["stats"]["std"]))
        logging.debug("%s, %.2f: M=%.3f std=%.3f" % (ket, kv, v["stats"]["mean"], v["stats"]["std"]))

#
#             max_dir_count = max(max_dir_count, len(rose.dom_dirs))
#             logging.debug(rose.dom_dirs)
#
#     ref_orientations = next(item['directions'] for item in batch['variations'] if (
#             item['ang_error'] == 0.0 and item['obs_count'] == 0.0 and item['line_err'] == 0.0 and item[
#         'laser_noise'] == 0.0))
#
#     error_matrix = np.array([[-1.] * max_dir_count for i in range(len(ang_error_variations))])
#     # error_matrix = np.array((len(ang_error_variations), max_dir_count), dtype=float)
#     # error_matrix.fill(-1.)
#     for i, err in enumerate(ang_error_variations):
#         test_orientations = next(item['directions'] for item in batch['variations'] if item['ang_error'] == err)
#         for j, to in enumerate(test_orientations):
#
#             min_err = np.pi
#             for ro in ref_orientations:
#                 min_err = min(min_err, pairwise_distance(ro, to))
#
#             error_matrix[i, j] = min_err
#     batch['error_matrix'] = error_matrix
#     all_error_matrix.append(error_matrix)
#
# cum_err = []
# cum_err_ampl = {1.0: [],
#                 5.0: [],
#                 10.0: [],
#                 15.0: [],
#                 20.0: [],
#                 30.0: [],
#                 40.0: [],
#                 45.0: [],
#                 75.0: [],
#                 90.0: []}
# for err_m in all_error_matrix:
#     cum_err.extend(list(filter((-1.).__ne__, list(err_m.flatten()))))
#
#     cum_err_ampl[1.0].extend(list(filter((-1.).__ne__, list(err_m[0]))))
#     cum_err_ampl[5.0].extend(list(filter((-1.).__ne__, list(err_m[1]))))
#     cum_err_ampl[10.0].extend(list(filter((-1.).__ne__, list(err_m[2]))))
#     cum_err_ampl[15.0].extend(list(filter((-1.).__ne__, list(err_m[3]))))
#     cum_err_ampl[20.0].extend(list(filter((-1.).__ne__, list(err_m[4]))))
#     cum_err_ampl[30.0].extend(list(filter((-1.).__ne__, list(err_m[5]))))
#     cum_err_ampl[40.0].extend(list(filter((-1.).__ne__, list(err_m[6]))))
#     cum_err_ampl[45.0].extend(list(filter((-1.).__ne__, list(err_m[7]))))
#     cum_err_ampl[75.0].extend(list(filter((-1.).__ne__, list(err_m[8]))))
#     cum_err_ampl[90.0].extend(list(filter((-1.).__ne__, list(err_m[9]))))
#
# # compute statistics
# mean_whole = stat.mean(cum_err)
# mean_per_ampl = {1.0: stat.mean(cum_err_ampl[1.0]),
#                  5.0: stat.mean(cum_err_ampl[5.0]),
#                  10.0: stat.mean(cum_err_ampl[10.0]),
#                  15.0: stat.mean(cum_err_ampl[15.0]),
#                  20.0: stat.mean(cum_err_ampl[20.0]),
#                  30.0: stat.mean(cum_err_ampl[30.0]),
#                  40.0: stat.mean(cum_err_ampl[40.0]),
#                  45.0: stat.mean(cum_err_ampl[45.0]),
#                  75.0: stat.mean(cum_err_ampl[75.0]),
#                  90.0: stat.mean(cum_err_ampl[90.0])}
#
# std_whole = stat.stdev(cum_err)
#
# std_per_ampl = {1.0: stat.stdev(cum_err_ampl[1.0]),
#                 5.0: stat.stdev(cum_err_ampl[5.0]),
#                 10.0: stat.stdev(cum_err_ampl[10.0]),
#                 15.0: stat.stdev(cum_err_ampl[15.0]),
#                 20.0: stat.stdev(cum_err_ampl[20.0]),
#                 30.0: stat.stdev(cum_err_ampl[30.0]),
#                 40.0: stat.stdev(cum_err_ampl[40.0]),
#                 45.0: stat.stdev(cum_err_ampl[45.0]),
#                 75.0: stat.stdev(cum_err_ampl[75.0]),
#                 90.0: stat.stdev(cum_err_ampl[90.0]),
#                 }
#
# logging.debug('error statistics whole')
# logging.debug(mean_whole)
# logging.debug(std_whole)
#
# logging.debug('error statistics per angle')
# logging.debug(mean_per_ampl)
# logging.debug(std_per_ampl)
#
# print("whole population - mean: {:.2f} std: {:.2f}".format(mean_whole, std_whole))
#
# for ang in ang_error_variations:
#     print("{:.1f} - mean: {:.2f} std: {:.2f}".format(ang, mean_per_ampl[ang], std_per_ampl[ang]))
