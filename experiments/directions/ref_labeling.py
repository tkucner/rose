from os import listdir
from os.path import isfile, join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

house_expo_dir = "/home/tzkr/python_workspace/HouseExpo/experiments/map_id_100/"
mapfiles = [f for f in listdir(house_expo_dir) if isfile(join(house_expo_dir, f))]

ids = []
for mf_full in mapfiles:
    mf = mf_full[:-4]
    split_mf = mf.split("_")
    ids.append(split_mf[0])
ids = list(set(ids))
ids.sort()

ref_map_label = '_obs_num_0.0_slamErr_linear_0.0_slamErr_angular_0.0_laser_noise_0.png'

for m in ids:
    ref_map_path = join(house_expo_dir, m + ref_map_label)
    img = mpimg.imread(ref_map_path)
    imgplot = plt.imshow(img)
    plt.show()
