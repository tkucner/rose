import argparse
import logging
from datetime import datetime

from skimage import io
from skimage.util import img_as_ubyte

import helpers as he
from extended_validator import ExtendedValidator
from fft_structure_extraction import FFTStructureExtraction as structure_extraction
from visualisation import visualisation

# time
now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(filename='logs/rose_' + date_time + '.log', level=logging.DEBUG,
                    format='%(levelname)s:%(module)s:%(lineno)d:%(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='JSON configuration file')
parser.add_argument('schema_file', help='JSON schema file')
args = parser.parse_args()
# Logging
logging.debug("config file: %s", args.config_file)
logging.debug("schema file: %s", args.schema_file)

json_validation = ExtendedValidator(args.config_file, args.schema_file)

success, config = json_validation.extended_validator()
logging.debug("configuration: %s", config)
if not success:
    he.eprint(config)
    exit(-1)

# FFT

grid_map = img_as_ubyte(io.imread(config["input_map"]))
rose = structure_extraction(grid_map, peak_height=config["peak_extraction_parameters"]["peak_height"],
                            smooth=config["peak_extraction_parameters"]["smooth_histogram"],
                            sigma=config["peak_extraction_parameters"]["sigma"])
rose.process_map()

filter_level = config["filtering_parameters"]["filter_level"]

rose.simple_filter_map(filter_level)

rose.generate_initial_hypothesis(type='simple', min_wall=5)

rose.find_walls_flood_filing()

plots = visualisation(rose)
plots.show(config["visualisation_flags"])
