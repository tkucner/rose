import argparse
import logging
from datetime import datetime

from skimage import io
from skimage.util import img_as_ubyte

import helpers as he
from fft_filtering import FFTFiltering as filter
from helpers import extended_validator

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

success, config = extended_validator(args.config_file, args.schema_file)
logging.debug("configuration: %s", config)
if not success:
    he.eprint(config)
    exit(-1)

grid_map = img_as_ubyte(io.imread(config["input_map"]))
rose = filter(grid_map, **config["fft_filtering"])
rose.process_map()