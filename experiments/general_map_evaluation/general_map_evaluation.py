import argparse
import logging
import statistics as stat
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from sklearn.cluster import MeanShift

import helpers as he
from extended_validator import ExtendedValidator
from fft_structure_extraction import FFTStructureExtraction as FFT_se
from visualisation import visualisation


def MS_wrap(amp):
    amp = np.array(amp).reshape(-1, 1)
    clustering = MeanShift().fit(amp)
    return np.unique(clustering.labels_)


def reshape(list1, list2):
    last = 0
    res = []
    for ele in list1:
        res.append(list2[last: last + len(ele)])
        last += len(ele)
    return res


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

list_template = [[i - 1, i] for i in range(2, len(mapfiles) + 2, 2)]
mapfiles = reshape(list_template, mapfiles)

for map_set in mapfiles:
    grid_map_bad = img_as_ubyte(io.imread(join(house_expo_dir, map_set[0])))

    rose_bad = FFT_se(grid_map_bad, peak_height=config["peak_extraction_parameters"]["peak_height"],
                      smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                      sigma=config["peak_extraction_parameters"]["sigma"])
    rose_bad.process_map()

    grid_map_good = img_as_ubyte(io.imread(join(house_expo_dir, map_set[1])))

    rose_good = FFT_se(grid_map_good, peak_height=config["peak_extraction_parameters"]["peak_height"],
                       smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                       sigma=config["peak_extraction_parameters"]["sigma"])
    rose_good.process_map()

    plots_bad = visualisation(rose_bad)
    plots_bad.show(config["visualisation_flags"], map_set[0])

    plots_good = visualisation(rose_good)
    plots_good.show(config["visualisation_flags"], map_set[1])

    grounded_rose_bad = rose_bad.pol_h - min(rose_bad.pol_h)
    streached_rose_bad = grounded_rose_bad / max(grounded_rose_bad)
    signal_av_bad = stat.mean(streached_rose_bad)
    peak_av_bad = stat.mean(streached_rose_bad[rose_bad.peak_indices])

    grounded_rose_good = rose_good.pol_h - min(rose_good.pol_h)
    streached_rose_good = grounded_rose_good / max(grounded_rose_good)
    signal_av_good = stat.mean(streached_rose_good)
    peak_av_good = stat.mean(streached_rose_good[rose_good.peak_indices])
    print(map_set[0].split('.')[0])
    print("Good map: average signal: {:.2f}, average peak: {:.2f}, ratio: {:.2f}, clusters: {}".format(signal_av_good,
                                                                                                       peak_av_good,
                                                                                                       signal_av_good / peak_av_good,
                                                                                                       MS_wrap(
                                                                                                           streached_rose_good)))
    print("Bad map: average signal: {:.2f}, average peak: {:.2f}, ratio: {:.2f}, clusters: {} ".format(signal_av_bad,
                                                                                                       peak_av_bad,
                                                                                                       signal_av_bad / peak_av_bad,
                                                                                                       MS_wrap(
                                                                                                           streached_rose_bad)))
