import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def file_path_formatter(directory, file):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    map_file_name_directory = os.path.splitext(os.path.basename(file))[0] + "_" + dt_string
    save_directory = os.path.join(directory, map_file_name_directory)
    if os.path.isdir(save_directory):
        print("Directory exits, data will be over written")
    else:
        os.makedirs(save_directory)
    return save_directory


class Visualisation:
    def __init__(self, structure, flags, source_file=None):

        self.structure = structure
        self.flags = flags
        self.save_dir = None
        if not self.flags["Save path"] == "" and source_file is not None:
            self.save_dir = file_path_formatter(self.flags["Save path"], source_file)

    def __save_plot(self, name):
        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, name.replace(" ", "_") + ".pdf"), bbox_inches='tight')

    def show(self):

        if self.flags["Input map"]:
            name = "Input map"
            plt.figure()
            plt.imshow(self.structure.grid_map, cmap="gray")
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Binary map"]:
            name = "Binary map"
            plt.figure()
            plt.imshow(self.structure.binary_map, cmap="gray")
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Frequency image"]:
            name = "Frequency image"
            plt.figure()
            plt.imshow(np.abs(self.structure.frequency_image), cmap="nipy_spectral")
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Frequency image with filter"]:
            name = "Frequency image with filter"
            plt.figure()
            plt.imshow(np.abs(self.structure.frequency_image), cmap="nipy_spectral")
            plt.imshow(np.logical_not(self.structure.filter) * 1, cmap='gray', alpha=0.25)
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Frequency image with dominant directions"]:
            name = "Frequency image with dominant directions"
            plt.figure()
            plt.imshow(np.abs(self.structure.frequency_image), cmap="nipy_spectral")
            for l in self.structure.lines:
                plt.plot([l[1], l[3]], [l[0], l[2]])
            plt.xlim(0, self.structure.frequency_image.shape[1])
            plt.ylim(0, self.structure.frequency_image.shape[0])
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Filter"]:
            name = "Filter"
            plt.figure()
            plt.imshow(np.abs(self.structure.filter), cmap="gray")
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Filtered frequency image"]:
            name = "Filtered frequency image"
            plt.figure()
            plt.imshow(np.abs(self.structure.filtered_frequency_image), cmap="nipy_spectral")
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Polar frequency image"]:
            name = "Polar frequency image"
            plt.figure()
            plt.imshow(np.flipud(self.structure.polar_frequency_image), aspect='auto', extent=(
                np.min(self.structure.angles), np.max(self.structure.angles), 0,
                np.max(self.structure.discretised_radius)), cmap="nipy_spectral")
            plt.xlim([np.min(self.structure.angles), np.max(self.structure.angles)])
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Polar frequency histogram"]:
            name = "Polar frequency histogram"
            plt.figure()
            plt.plot(self.structure.angles, self.structure.polar_amplitude_histogram, '.c')
            plt.plot(self.structure.angles[self.structure.peak_indices],
                     self.structure.polar_amplitude_histogram[self.structure.peak_indices], 'r+')
            for p in self.structure.peak_pairs:
                plt.scatter(self.structure.angles[p], self.structure.polar_amplitude_histogram[p], marker='^', s=120)
            plt.title(name)
            self.__save_plot(name)

        if self.flags["Polar frequency image and histogram"]:
            name = "Polar frequency image and histogram"
            plt.figure()
            plt.imshow(np.flipud(self.structure.polar_frequency_image), cmap="nipy_spectral", aspect='auto',
                       extent=(np.min(self.structure.angles), np.max(self.structure.angles), 0,
                               np.max(self.structure.discretised_radius)))
            plt.xlim([np.min(self.structure.angles), np.max(self.structure.angles)])
            plt.xlabel("Orientation [rad]")
            plt.ylabel("Radius in pixel")
            ax2 = plt.twinx()
            ax2.plot(self.structure.angles, self.structure.polar_amplitude_histogram, '.c')
            ax2.plot(self.structure.angles[self.structure.peak_indices],
                     self.structure.polar_amplitude_histogram[self.structure.peak_indices], 'r+')
            for p in self.structure.peak_pairs:
                ax2.scatter(self.structure.angles[p], self.structure.polar_amplitude_histogram[p], marker='^', s=120)
            ax2.set_ylabel("Orientation score")

            plt.title(name)
            self.__save_plot(name)

    plt.show()