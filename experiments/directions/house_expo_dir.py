import argparse
import logging
import pickle
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
from fft_structure_extraction import FFTStructureExtraction as FFT_se


# Logging
# time

def pairwise_distance(pair1, pair2):
    dist = []
    for x1 in pair1:
        for x2 in pair2:
            x1 = x1 if x1 < np.pi else x1 - np.pi
            x2 = x2 if x2 < np.pi else x2 - np.pi
            dist.append(min_wrapped_distance(x1, x2))
    return min(dist)


def min_wrapped_distance(x1, x2):
    dx1 = abs(x1 - x2)
    dx2 = np.pi
    if dx1 > np.pi / 2:
        dx2 = np.pi - dx1
    return min(dx1, dx2)


tqdm_falg = False

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

    map_identification = {'id': split_mf[0], 'file_name': mf_full, error_names[0]: float(split_mf[3]),
                          error_names[1]: float(split_mf[6]), error_names[2]: float(split_mf[9]),
                          error_names[3]: float(split_mf[12])}

    for en in error_names:
        if map_identification[en] != 0.0:
            error_types[en].add(map_identification[en])

    logging.debug("found map: %s", mf)
    if map_identification['id'] in [x['id'] for x in batches]:
        it = next(item for item in batches if item["id"] == map_identification['id'])
        it['variations'].append(map_identification)
        logging.debug("updating batch: %s", map_identification['id'])
    else:
        lb = {"id": map_identification['id'], 'variations': [map_identification]}
        batches.append(lb)
        logging.debug("new batch: %s", map_identification['id'])
for en in error_names:
    error_types[en] = list(error_types[en])
    error_types[en].sort()

for et_key in error_types:
    error_types[et_key] = {
        i: {"ang_errors": [], "directions_count_error": [], "stats_ang": {}, "stats_directions_count": {}} for i in
        error_types[et_key]}

# computing angles
all_error_matrix = []
max_dir_count = 0
for batch in tqdm(batches, desc="Processed environments", disable=tqdm_falg):
    ref_map = []
    # compute orientations
    for variant in tqdm(batch['variations'], desc="finding directions " + batch['id'], disable=tqdm_falg):
        grid_map = img_as_ubyte(io.imread(join(house_expo_dir, variant['file_name'])))
        rose = FFT_se(grid_map, peak_height=config["peak_extraction_parameters"]["peak_height"],
                      smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                      sigma=config["peak_extraction_parameters"]["sigma"])
        rose.process_map()
        variant['directions'] = rose.dom_dirs
        # find reference for the variant
        if variant[error_names[0]] == 0.0 and variant[error_names[1]] == 0.0 and variant[error_names[2]] == 0.0 and \
                variant[error_names[3]] == 0.0:
            ref_map = rose.dom_dirs
    # compute errors
    for variant in tqdm(batch['variations'], desc="computing error " + batch['id'], disable=tqdm_falg):
        variant['errors'] = []
        variant['directions_count_error'] = []
        if not (variant[error_names[0]] == 0.0 and variant[error_names[1]] == 0.0 and variant[error_names[2]] == 0.0 and
                variant[error_names[3]] == 0.0):
            for d in variant['directions']:
                min_err = np.pi
                for r in ref_map:
                    min_err = min(min_err, pairwise_distance(d, r))
                variant['errors'].append(min_err)
            variant['directions_count_error'].append(abs(len(ref_map) - len(variant['directions'])))
            for en in error_names:
                if variant[en] != 0.0:
                    error_types[en][variant[en]]["ang_errors"].extend(variant['errors'])
                    error_types[en][variant[en]]["directions_count_error"].extend(variant['directions_count_error'])

# compute statistics
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

f = open("results/dir_error_types_FFT.pkl", "wb")
pickle.dump(error_types, f)
f.close()

f = open("results/dir_batches_FFT.pkl", "wb")
pickle.dump(batches, f)
f.close()
