import argparse
import logging
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
from joblib import Parallel, delayed
from skimage import io
from skimage.filters import threshold_yen
from skimage.util import img_as_ubyte

import helpers as he
from extended_validator import ExtendedValidator
from fft_structure_extraction import FFTStructureExtraction as FFT_se
from visualisation import visualisation


def load_map(grid_map):
    if len(grid_map.shape) == 3:
        grid_map = grid_map[:, :, 1]
    thresh = threshold_yen(grid_map)
    binary_map = grid_map <= thresh
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
evaluate = True
visualise = False
reference_dir = config["reference_dir"]

# 1 -> get the reference maps
reference_map_files = [f for f in listdir(reference_dir) if isfile(join(reference_dir, f)) and "good" in f]
reference_maps = {}

for rmf in reference_map_files:
    grid_map = img_as_ubyte(io.imread(join(reference_dir, rmf)))
    ref_map = load_map(grid_map)
    reference_maps[rmf.split(".")[0]] = ref_map


def parallel_wrapper(map_set):
    filter_levels = list(np.arange(0, 1.05, 0.05))
    result_file = map_set.split(".")[0]
    result_file = result_file + ".txt"

    for fl in filter_levels:
        f = open(join("results", result_file), 'a+')
        if evaluate:
            grid_map = img_as_ubyte(io.imread(join(house_expo_dir, map_set)))

            rose = FFT_se(grid_map, peak_height=config["peak_extraction_parameters"]["peak_height"],
                          smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                          sigma=config["peak_extraction_parameters"]["sigma"],
                          par=config["peak_extraction_parameters"]["par"])
            rose.process_map()
            filter_level = config["filtering_parameters"]["filter_level"]

            rose.simple_filter_map(fl)
            # rose.histogram_filtering()

        if visualise:
            plots = visualisation(rose)
            plots.show(config["visualisation_flags"])

        # compute precision recall
        # 1 -> find reference map
        name = map_set.split("_ocount")[0]
        ref_map = reference_maps[name]
        # 2 -> count true positives
        tp = np.logical_and(ref_map, rose.analysed_map)

        noise = np.logical_xor(rose.binary_map, ref_map)
        noise_labels = np.logical_xor(rose.analysed_map, rose.binary_map)

        # 3-> find true negatives
        tn = np.logical_and(noise, noise_labels)

        # 4-> find false negatives
        fn = np.logical_xor(noise_labels, tn)

        # 5-> find false positive
        fp = np.logical_xor(tp, rose.analysed_map)

        # compute counts
        ctp = np.sum(tp)
        ctn = np.sum(tn)
        cfn = np.sum(fn)
        cfp = np.sum(fp)

        precision = ctp / (ctp + cfp)
        recall = ctp / (ctp + cfn)
        true_negative_rate = ctn / (ctn + cfp)
        # logging.debug(
        #     "%s threshold:%.3f precision:%.3f recall:%.3f true_negative_rate:%.3f true_positive:%d true_negative:%d false_positive:%d, false_negative:%d",
        #     map_set,
        #     rose.quality_threshold, precision, recall, true_negative_rate, ctp, ctn, cfp, cfn)
        f.write(
            "{} threshold:{:.3f} precision:{:.3f} recall:{:.3f} true_negative_rate:{:.3f} true_positive:{:d} true_negative:{:d} false_positive:{:d} false_negative:{:d} \n".format(
                map_set,
                rose.quality_threshold, precision, recall, true_negative_rate, ctp, ctn, cfp, cfn))
        print(
            "{} threshold:{:.3f} precision:{:.3f} recall:{:.3f} true_negative_rate:{:.3f} true_positive:{:d} true_negative:{:d} false_positive:{:d} false_negative:{:d}".format(
                map_set,
                rose.quality_threshold, precision, recall, true_negative_rate, ctp, ctn, cfp, cfn))
        f.close()


# for map_set in mapfiles:
Parallel(n_jobs=5)(delayed(parallel_wrapper)(map_set) for map_set in mapfiles)
