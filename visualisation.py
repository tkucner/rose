import logging

import matplotlib.pyplot as plt
import numpy as np

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class visualisation:
    def __init__(self, structure, flags):

        self.structure = structure
        self.flags = flags

    def show(self):

        if self.flags["Input map"]:
            plt.figure()
            plt.imshow(self.structure.grid_map)
            plt.title("Raw map")

        if self.flags["Binary map"]:
            plt.figure()
            plt.imshow(self.structure.binary_map)
            plt.title("Binarised map")

        if self.flags["Frequency image"]:
            plt.figure()
            plt.imshow(np.abs(self.structure.frequency_image))
            plt.title("Frequency image of the map")

        if self.flags["Frequency image"]:
            plt.figure()
            plt.imshow(np.abs(self.structure.frequency_image))
            plt.title("Frequency image of the map")

        if self.flags["Frequency image with dominant directions"]:
            plt.figure()
            plt.imshow(np.abs(self.structure.frequency_image))
            for l in self.structure.lines:
                plt.plot([l[1], l[3]], [l[0], l[2]])
            plt.xlim(0, self.structure.frequency_image.shape[1])
            plt.ylim(0, self.structure.frequency_image.shape[0])
            plt.title("Frequency image of the map with dominant directions")

        if self.flags["Filter"]:
            plt.figure()
            plt.imshow(np.abs(self.structure.filter))
            plt.title("Filter")

        if self.flags["Polar frequency image"]:
            plt.figure()
            plt.imshow(np.flipud(self.structure.polar_frequency_image))
            plt.title("Polar frequency image of the map")

    plt.show()
