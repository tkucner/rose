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
for mf_full in mapfiles:
    mf = mf_full[:-4]
    split_mf = mf.split("_")
    map_identification = {'id': split_mf[0], "obs_count": float(split_mf[3]), "line_err": float(split_mf[6]),
                          "ang_error": float(split_mf[9]), "laser_noise": float(split_mf[12]), 'file_name': mf_full}
    logging.debug("found map: %s", mf)
    if map_identification['id'] in [x['id'] for x in batches]:
        it = next(item for item in batches if item["id"] == map_identification['id'])
        it['variations'].append(map_identification)
        logging.debug("updating batch: %s", map_identification['id'])
    else:
        lb = {"id": map_identification['id'], 'variations': [map_identification]}
        batches.append(lb)
        logging.debug("new batch: %s", map_identification['id'])

# errors variants
ref_ang = 0.0
ang_error_variations = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 45.0, 75.0]
# computing angles
all_error_matrix = []
for batch in tqdm(batches):
    logging.debug("--------------------")
    for variant in batch['variations']:
        grid_map = img_as_ubyte(io.imread(join(house_expo_dir, variant['file_name'])))
        rose = structure_extraction(grid_map, peak_height=config["peak_extraction_parameters"]["peak_height"],
                                    smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                                    sigma=config["peak_extraction_parameters"]["sigma"])
        rose.process_map()
        variant['directions'] = rose.dom_dirs

        logging.debug(rose.dom_dirs)

    # analise variants
    ref_orientations = next(item['directions'] for item in batch['variations'] if item['ang_error'] == ref_ang)
    error_matrix = np.zeros((len(ang_error_variations), len(ref_orientations)), dtype=float)
    for i, err in enumerate(ang_error_variations):
        test_orientations = next(item['directions'] for item in batch['variations'] if item['ang_error'] == err)
        for j, ro in enumerate(ref_orientations):
            min_err = np.pi
            for to in test_orientations:
                min_err = min(min_err, pairwise_distance(ro, to))
            error_matrix[i, j] = min_err
    batch['error_matrix'] = error_matrix
    all_error_matrix.append(error_matrix)

cum_err = []
cum_err_ampl = {1.0: [],
                5.0: [],
                10.0: [],
                15.0: [],
                20.0: [],
                30.0: [],
                40.0: [],
                45.0: [],
                75.0: []}
for err_m in all_error_matrix:
    cum_err.extend(list(error_matrix.flatten()))

    cum_err_ampl[1.0].extend(list(error_matrix[0]))
    cum_err_ampl[5.0].extend(list(error_matrix[1]))
    cum_err_ampl[10.0].extend(list(error_matrix[2]))
    cum_err_ampl[15.0].extend(list(error_matrix[3]))
    cum_err_ampl[20.0].extend(list(error_matrix[4]))
    cum_err_ampl[30.0].extend(list(error_matrix[5]))
    cum_err_ampl[40.0].extend(list(error_matrix[6]))
    cum_err_ampl[45.0].extend(list(error_matrix[7]))
    cum_err_ampl[75.0].extend(list(error_matrix[8]))

# compute statistics
mean_whole = stat.mean(cum_err)
mean_per_ampl = {1.0: stat.mean(cum_err_ampl[1.0]),
                 5.0: stat.mean(cum_err_ampl[5.0]),
                 10.0: stat.mean(cum_err_ampl[10.0]),
                 15.0: stat.mean(cum_err_ampl[15.0]),
                 20.0: stat.mean(cum_err_ampl[20.0]),
                 30.0: stat.mean(cum_err_ampl[30.0]),
                 40.0: stat.mean(cum_err_ampl[40.0]),
                 45.0: stat.mean(cum_err_ampl[45.0]),
                 75.0: stat.mean(cum_err_ampl[75.0])}

std_whole = stat.stdev(cum_err)

std_per_ampl = {1.0: stat.stdev(cum_err_ampl[1.0]),
                5.0: stat.stdev(cum_err_ampl[5.0]),
                10.0: stat.stdev(cum_err_ampl[10.0]),
                15.0: stat.stdev(cum_err_ampl[15.0]),
                20.0: stat.stdev(cum_err_ampl[20.0]),
                30.0: stat.stdev(cum_err_ampl[30.0]),
                40.0: stat.stdev(cum_err_ampl[40.0]),
                45.0: stat.stdev(cum_err_ampl[45.0]),
                75.0: stat.stdev(cum_err_ampl[75.0])}

logging.debug('error statistics whole')
logging.debug(mean_whole)
logging.debug(std_whole)

logging.debug('error statistics per angle')
logging.debug(mean_per_ampl)
logging.debug(std_per_ampl)

print("whole population - mean: {:.2f} std: {:.2f}".format(mean_whole, std_whole))

for ang in ang_error_variations:
    print("{:.1f} - mean: {:.2f} std: {:.2f}".format(ang, mean_per_ampl[ang], std_per_ampl[ang]))
