import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import helpers

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def save_plot(save_dir, name):
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name.replace(" ", "_") + ".pdf"), bbox_inches='tight')


class VisualisationFFT:
    def __init__(self, fft_filtered, flags, source_file=None):

        self.fft_filtered = fft_filtered
        self.flags = flags
        self.save_dir = None
        if not self.flags["Save path"] == "" and source_file is not None:
            self.save_dir = helpers.file_path_formatter(self.flags["Save path"], source_file)

    def show(self):

        if self.flags["Input map"]:
            name = "Input map"
            plt.figure()
            plt.imshow(self.fft_filtered.grid_map, cmap="gray")
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Binary map"]:
            name = "Binary map"
            plt.figure()
            plt.imshow(self.fft_filtered.binary_map, cmap="gray")
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Frequency image"]:
            name = "Frequency image"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.frequency_image), cmap="nipy_spectral")
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Frequency image with filter"]:
            name = "Frequency image with filter"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.frequency_image), cmap="nipy_spectral")
            plt.imshow(np.logical_not(self.fft_filtered.filter) * 1, cmap='gray', alpha=0.25)
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Frequency image with dominant directions"]:
            name = "Frequency image with dominant directions"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.frequency_image), cmap="nipy_spectral")
            for l in self.fft_filtered.lines:
                plt.plot([l.coords[0][1], l.coords[1][1]], [l.coords[0][0], l.coords[1][0]])
            plt.xlim(0, self.fft_filtered.frequency_image.shape[1])
            plt.ylim(0, self.fft_filtered.frequency_image.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Filter"]:
            name = "Filter"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.filter), cmap="gray")
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Filtered frequency image"]:
            name = "Filtered frequency image"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.filtered_frequency_image), cmap="nipy_spectral")
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Polar frequency image"]:
            name = "Polar frequency image"
            plt.figure()
            plt.imshow(np.flipud(self.fft_filtered.polar_frequency_image), aspect='auto', extent=(
                np.min(self.fft_filtered.angles), np.max(self.fft_filtered.angles), 0,
                np.max(self.fft_filtered.discretised_radius)), cmap="nipy_spectral")
            plt.xlim([np.min(self.fft_filtered.angles), np.max(self.fft_filtered.angles)])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Polar frequency histogram"]:
            name = "Polar frequency histogram"
            plt.figure()
            plt.plot(self.fft_filtered.angles, self.fft_filtered.polar_amplitude_histogram, '.c')
            plt.plot(self.fft_filtered.angles[self.fft_filtered.peak_indices],
                     self.fft_filtered.polar_amplitude_histogram[self.fft_filtered.peak_indices], 'r+')
            for p in self.fft_filtered.peak_pairs:
                plt.scatter(self.fft_filtered.angles[p], self.fft_filtered.polar_amplitude_histogram[p], marker='^',
                            s=120)
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Polar frequency image and histogram"]:
            name = "Polar frequency image and histogram"
            plt.figure()
            plt.imshow(np.flipud(self.fft_filtered.polar_frequency_image), cmap="nipy_spectral", aspect='auto',
                       extent=(np.min(self.fft_filtered.angles), np.max(self.fft_filtered.angles), 0,
                               np.max(self.fft_filtered.discretised_radius)))
            plt.xlim([np.min(self.fft_filtered.angles), np.max(self.fft_filtered.angles)])
            plt.xlabel("Orientation [rad]")
            plt.ylabel("Radius in pixel")
            ax2 = plt.twinx()
            ax2.plot(self.fft_filtered.angles, self.fft_filtered.polar_amplitude_histogram, '.c')
            ax2.plot(self.fft_filtered.angles[self.fft_filtered.peak_indices],
                     self.fft_filtered.polar_amplitude_histogram[self.fft_filtered.peak_indices], 'r+')
            for p in self.fft_filtered.peak_pairs:
                ax2.scatter(self.fft_filtered.angles[p], self.fft_filtered.polar_amplitude_histogram[p], marker='^',
                            s=120)
            ax2.set_ylabel("Orientation score")

            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

            if self.flags["Reconstructed map"]:
                name = "Reconstructed map"
                plt.figure()
                plt.imshow(np.abs(self.fft_filtered.reconstructed_map), cmap="nipy_spectral")
                plt.title(name)
                if self.flags["Save path"] != "":
                    save_plot(self.save_dir, name)

            if self.flags["Scored map"]:
                name = "Scored map"
                plt.figure()
                plt.imshow(np.abs(self.fft_filtered.map_scored), cmap="nipy_spectral")
                plt.colorbar()
                plt.title(name)
                if self.flags["Save path"] != "":
                    save_plot(self.save_dir, name)

            if self.flags["Filtered map overlay"]:
                name = "Filtered map overlay (" + str(self.fft_filtered.quality_threshold) + ")"
                plt.figure()
                plt.imshow(np.abs(self.fft_filtered.binary_map), cmap="gray")
                plt.imshow(np.logical_not(self.fft_filtered.analysed_map) * 1, cmap='gray', alpha=0.75)

                plt.title(name)
                if self.flags["Save path"] != "":
                    save_plot(self.save_dir, name)

            if self.flags["Filtered map final"]:
                name = "Filtered map final (" + str(self.fft_filtered.quality_threshold) + ")"
                plt.figure()
                plt.imshow(np.logical_not(self.fft_filtered.analysed_map) * 1, cmap='gray')

                plt.title(name)
                if self.flags["Save path"] != "":
                    save_plot(self.save_dir, name)

            if self.flags["Cell histogram"] and not self.fft_filtered.pixel_quality_histogram == []:
                print(self.fft_filtered.pixel_quality_histogram == [])
                name = "Cell histogram"
                plt.figure()
                x = np.arange(np.min(self.fft_filtered.pixel_quality_histogram["edges"]),
                              np.max(self.fft_filtered.pixel_quality_histogram["edges"]),
                              (np.max(self.fft_filtered.pixel_quality_histogram["edges"]) - np.min(
                                  self.fft_filtered.pixel_quality_histogram["edges"])) / 1000)

                if self.fft_filtered.pixel_quality_gmm["means"][0] < self.fft_filtered.pixel_quality_gmm["means"][1]:
                    y_b = stats.norm.pdf(x, self.fft_filtered.pixel_quality_gmm["means"][0],
                                         math.sqrt(self.fft_filtered.pixel_quality_gmm["covariances"][0])) * \
                          self.fft_filtered.pixel_quality_gmm["weights"][0]
                    y_g = stats.norm.pdf(x, self.fft_filtered.pixel_quality_gmm["means"][1],
                                         math.sqrt(self.fft_filtered.pixel_quality_gmm["covariances"][1])) * \
                          self.fft_filtered.pixel_quality_gmm["weights"][1]
                else:
                    y_g = stats.norm.pdf(x, self.fft_filtered.pixel_quality_gmm["means"][0],
                                         math.sqrt(self.fft_filtered.pixel_quality_gmm["covariances"][0])) * \
                          self.fft_filtered.pixel_quality_gmm["weights"][0]
                    y_b = stats.norm.pdf(x, self.fft_filtered.pixel_quality_gmm["means"][1],
                                         math.sqrt(self.fft_filtered.pixel_quality_gmm["covariances"][1])) * \
                          self.fft_filtered.pixel_quality_gmm["weights"][1]

                plt.bar(self.fft_filtered.pixel_quality_histogram["centers"],
                        self.fft_filtered.pixel_quality_histogram["bins"],
                        width=self.fft_filtered.pixel_quality_histogram["width"])
                plt.plot(x, y_b, 'r')
                plt.plot(x, y_g, 'g')
                plt.axvline(x=self.fft_filtered.quality_threshold, color='y')
                plt.title(name)
                if self.flags["Save path"] != "":
                    save_plot(self.save_dir, name)

        if self.flags["Show plots"]:
            plt.show()


class VisualisationStructure:
    def __init__(self, structured_map, flags, source_file=None):
        self.structured_map = structured_map
        self.flags = flags
        self.save_dir = None
        if not self.flags["Save path"] == "" and source_file is not None:
            self.save_dir = helpers.file_path_formatter(self.flags["Save path"], source_file)

    def show(self):

        if self.flags["Map with directions"]:
            name = "Map with directions"
            plt.figure()
            plt.imshow(self.structured_map.fft_map.binary_map, cmap="gray")
            for l in self.structured_map.dominant_lines:
                plt.plot([l.coords[0][1], l.coords[1][1]], [l.coords[0][0], l.coords[1][0]])
            plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
            plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Show plots"]:
            plt.show()
