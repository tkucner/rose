import logging
import math
import statistics as stat
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import helpers as he

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# output
###########################
class visualisation:
    def __init__(self, structure):
        """
        Args:
            structure:
        """
        self.structure = structure

    def __show_patches(self, ax):
        """
        Args:
            ax:
        """
        ax.imshow(self.structure.binary_map, cmap="gray")
        non_zero_ind = np.nonzero(self.structure.analysed_map)
        for ind in zip(non_zero_ind[0], non_zero_ind[1]):
            square = patches.Rectangle((ind[1], ind[0]), 1, 1, color='green')
            ax.add_patch(square)
        return ax

    # def __show_labels(self, ax):
    #     cmap = plt.cm.get_cmap("tab10")
    #     cmap.set_under("black")
    #     cmap.set_over("yellow")
    #
    #     ax.imshow(self.structure.labeled_map, cmap=cmap, vmin=1)
    #     return ax

    @staticmethod
    def __show_short_segments(ax, segments, segments_mbb_lines):
        """
        Args:
            ax:
            segments:
            segments_mbb_lines:
        """
        for local_segments, local_mbb_lines in zip(segments, segments_mbb_lines):
            for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
                ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
                wall = he.cetral_line(l_segment.minimal_bounding_box)
                ax.plot([wall[0].y, wall[1].y], [wall[0].x, wall[1].x], 'c')
                # if l_segment.mbb_area > 10:
                # ax.plot([l_mbb_lines["Y1"], l_mbb_lines["Y2"]], [l_mbb_lines["X1"], l_mbb_lines["X2"]], 'g')
        return ax

    @staticmethod
    def __show_long_segments(ax, segments, segments_mbb_lines):
        """
        Args:
            ax:
            segments:
            segments_mbb_lines:
        """
        for local_segments, local_mbb_lines in zip(segments, segments_mbb_lines):
            for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
                ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
                if l_segment.mbb_area > 10:
                    ax.plot([l_mbb_lines["Y1"], l_mbb_lines["Y2"]], [l_mbb_lines["X1"], l_mbb_lines["X2"]], 'g')
        return ax

    def show(self, visualisation_flags, name=[]):
        """
        Args:
            visualisation_flags:
            name:
        """
        t = time.time()
        if visualisation_flags["Binary map"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(self.structure.binary_map * 1, cmap="gray")
            ax.axis("off")
            name = "Binary Map"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["FFT Spectrum"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow((np.abs(self.structure.frequency_image)), cmap="nipy_spectral")
            ax.axis("off")
            name = "FFT Spectrum"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["FFT spectrum with directions"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow((np.abs(self.structure.frequency_image)), cmap="nipy_spectral")
            for l in self.structure.lines:
                ax.plot([l[1], l[3]], [l[0], l[2]])
            ax.axis("off")
            name = "FFT Spectrum with directions"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            ax.set_xlim(0, self.structure.frequency_image.shape[1])
            ax.set_ylim(0, self.structure.frequency_image.shape[0])
            plt.show()

        if visualisation_flags["Map with walls"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(self.structure.binary_map, cmap="gray")
            for indices_slice in self.structure.slices_v_ids:
                for indices in zip(indices_slice[0], indices_slice[1]):
                    for i in zip(indices[0], indices[1]):
                        ax.plot(i[1], i[0], 'rx')
            for indices_slice in self.structure.slices_h_ids:
                for indices in zip(indices_slice[0], indices_slice[1]):
                    for i in zip(indices[0], indices[1]):
                        ax.plot(i[1], i[0], 'rx')
            ax.axis("off")
            name = "Map with walls"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["Map with directions"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(self.structure.binary_map, cmap="gray")
            for line in self.structure.lines_hypothesis_v:
                ax.plot([line[0], line[2]], [line[3], line[1]], alpha=0.5)
            for line in self.structure.lines_hypothesis_h:
                ax.plot([line[0], line[2]], [line[3], line[1]], alpha=0.5)
            for cells, values in zip(self.structure.cell_hypothesis_v, self.structure.scored_hypothesis_v):
                ax.scatter(cells[1], cells[0], c='r', s=values * 100, alpha=0.5)
            for cells, values in zip(self.structure.cell_hypothesis_h, self.structure.scored_hypothesis_h):
                ax.scatter(cells[1], cells[0], c='r', s=values * 100, alpha=0.5)
            for cells, values in zip(self.structure.cell_hypothesis_v, self.structure.scored_hypothesis_v_cut):
                ax.scatter(cells[1], cells[0], c='g', s=values * 100, alpha=0.5)
            for cells, values in zip(self.structure.cell_hypothesis_h, self.structure.scored_hypothesis_h_cut):
                ax.scatter(cells[1], cells[0], c='g', s=values * 100, alpha=0.5)
            ax.axis("off")
            name = "Map with directions"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            ax.set_xlim(0, self.structure.binary_map.shape[1])
            ax.set_ylim(self.structure.binary_map.shape[0], 0)
            plt.show()

        if visualisation_flags["Unfolded FFT Spectrum"]:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(np.flipud(self.structure.polar_frequency_image), cmap="nipy_spectral", aspect='auto',
                      extent=(
                          np.min(self.structure.angles), np.max(self.structure.angles), 0,
                          np.max(self.structure.discretised_radius)))
            ax.set_xlim([np.min(self.structure.angles), np.max(self.structure.angles)])
            ax.set_xlabel("Orientation [rad]")
            ax.set_ylabel("Radius in pixel")
            ax2 = ax.twinx()
            ax2.plot(self.structure.angles, self.structure.polar_amplitude_histogram)
            ax2.plot(self.structure.angles[self.structure.peak_indices],
                     self.structure.polar_amplitude_histogram[self.structure.peak_indices], 'r+')
            for p in self.structure.peak_pairs:
                ax2.scatter(self.structure.angles[p], self.structure.polar_amplitude_histogram[p], marker='^', s=120)
            ax2.set_ylabel("Orientation score")
            name = "Unfolded FFT Spectrum"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["FFT Spectrum Signal"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow((np.abs(self.structure.filtered_frequency_image)), cmap="nipy_spectral")
            ax.axis("off")
            name = "FFT Spectrum Signal"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["FFT Spectrum Noise"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow((np.abs(self.structure.mask_inv_ft_image)), cmap="nipy_spectral")
            ax.axis("off")
            name = "FFT Spectrum Noise"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Map Scored Good"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(np.abs(self.structure.map_scored), cmap="plasma")
            ax.axis("off")
            name = "Map Scored Good"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Map Scored Bad"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(np.abs(self.structure.map_scored_bad), cmap="plasma")
            ax.axis("off")
            name = "Map Scored Bad"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Map Scored Diff"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(np.abs(self.structure.map_scored_diff), cmap="plasma")
            ax.axis("off")
            name = "Map Scored Diff"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Map Split Good"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(self.structure.map_split_good, cmap="plasma")
            ax.axis("off")
            name = "Map Split Good"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["FFT Map Split Good"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow((np.abs(self.structure.ft_image_split)), cmap="nipy_spectral")
            ax.axis("off")
            name = "FFT Map Split Good"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Side by Side"]:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
            ax[0].imshow(self.structure.binary_map * 1, cmap="gray")
            ax[0].axis("off")
            ax[1].imshow(np.abs(self.structure.map_scored), cmap="nipy_spectral")
            ax[1].axis("off")
            name1 = "Map"
            name2 = "Score"
            fig.canvas.set_window_title("Map Quality assessment")
            ax[0].set_title(name1)
            ax[1].set_title(name2)
            plt.tight_layout()
            plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0, wspace=0, hspace=0)
            plt.show()

        if visualisation_flags["Map and peaks"]:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(self.structure.binary_map * 1, cmap="gray")
            ax[0].axis("off")

            # signal_av=stat.mean((self.structure.pol_h-min(self.structure.pol_h))/max(self.structure.pol_h))
            # peak_av=stat.mean((self.structure.pol_h[self.structure.peak_indices]-min(self.structure.pol_h))/max(self.structure.pol_h))

            grounded_rose = self.structure.polar_amplitude_histogram - min(self.structure.polar_amplitude_histogram)
            streached_rose = grounded_rose / max(grounded_rose)
            signal_av = stat.mean(streached_rose)
            peak_av = stat.mean(streached_rose[self.structure.peak_indices])

            ax[1].plot(self.structure.angles, streached_rose)
            ax[1].plot(self.structure.angles[self.structure.peak_indices],
                       streached_rose[self.structure.peak_indices], 'r+')
            for p in self.structure.peak_pairs:
                ax[1].scatter(self.structure.angles[p], streached_rose[p], marker='^', s=120)

            ax[1].plot([self.structure.angles[0], self.structure.angles[-1]], [signal_av, signal_av])
            ax[1].plot([self.structure.angles[0], self.structure.angles[-1]], [peak_av, peak_av])

            ax[1].set_ylabel("Normalised orientation score")
            ax[1].set_ylim(0.0, 1.0)
            ax[1].set_xlabel("Orientation [rad]")

            name1 = "Map"
            name2 = "Unfolded FFT Spectrum"
            fig.canvas.set_window_title("Map Quality assessment")
            ax[0].set_title(name1)
            ax[1].set_title(name2)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(self.structure.angles, streached_rose)
            ax.plot(self.structure.angles[self.structure.peak_indices],
                    streached_rose[self.structure.peak_indices], 'r+')
            for p in self.structure.peak_pairs:
                ax.scatter(self.structure.angles[p], streached_rose[p], marker='^', s=120)

            ax.plot([self.structure.angles[0], self.structure.angles[-1]], [signal_av, signal_av])
            ax.plot([self.structure.angles[0], self.structure.angles[-1]], [peak_av, peak_av])

            ax.set_ylabel("Normalised orientation score")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Orientation [rad]")
            import tikzplotlib

            tikzplotlib.save(
                "/home/tzkr/python_workspace/rose/experiments/general_map_evaluation/tikz_plots/" + name.replace(".png",
                                                                                                                 ".tex"),
                figure=fig)

            plt.show()

        if visualisation_flags["Simple Filtered Map"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax = self.__show_patches(ax)
            # ax.imshow(self.structure.binary_map, cmap="gray")
            # non_zero_ind = np.nonzero(self.structure.analysed_map)
            # for ind in zip(non_zero_ind[0], non_zero_ind[1]):
            #     square = patches.Rectangle((ind[1], ind[0]), 1, 1, color='green')
            #     ax.add_patch(square)
            name = "Simple Filtered Map (" + str(self.structure.quality_threshold) + ")"
            ax.axis("off")
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Threshold Setup with Clusters"]:
            x = np.arange(np.min(self.structure.pixel_quality_histogram["edges"]),
                          np.max(self.structure.pixel_quality_histogram["edges"]),
                          (np.max(self.structure.pixel_quality_histogram["edges"]) - np.min(
                              self.structure.pixel_quality_histogram["edges"])) / 1000)

            if self.structure.pixel_quality_gmm["means"][0] < self.structure.pixel_quality_gmm["means"][1]:
                y_b = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][0],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][0])) * \
                      self.structure.pixel_quality_gmm["weights"][0]
                y_g = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][1],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][1])) * \
                      self.structure.pixel_quality_gmm["weights"][1]
            else:
                y_g = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][0],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][0])) * \
                      self.structure.pixel_quality_gmm["weights"][0]
                y_b = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][1],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][1])) * \
                      self.structure.pixel_quality_gmm["weights"][1]

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.bar(self.structure.pixel_quality_histogram["centers"], self.structure.pixel_quality_histogram["bins"],
                   width=self.structure.pixel_quality_histogram["width"])
            ax.plot(x, y_b, 'r')
            ax.plot(x, y_g, 'g')
            # ax.axvline(x=self.structure.cluster_quality_threshold, color='y')
            ax.axvline(x=self.structure.quality_threshold, color='y')
            name = "Treshold Setup with Clusters"
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Cluster Filtered Map"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax = self.__show_patches(ax)
            # ax.imshow(self.structure.binary_map, cmap="gray")
            # non_zero_ind = np.nonzero(self.structure.analysed_map)
            # for ind in zip(non_zero_ind[0], non_zero_ind[1]):
            #     square = patches.Rectangle((ind[1], ind[0]), 1, 1, color='green')
            #     ax.add_patch(square)
            # name = "Cluster Filtered Map (" + str(self.structure.cluster_quality_threshold) + ")"
            name = "Cluster Filtered Map (" + str(self.structure.quality_threshold) + ")"
            ax.axis("off")
            fig.canvas.set_window_title(name)
            ax.set_title(name)
            plt.show()

        if visualisation_flags["Map with slices"]:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(self.structure.binary_map, cmap="gray")
            for l in self.structure.slices_v:
                for s in zip(l[0], l[1]):
                    ax.plot(s[1], s[0], '.')
            for l in self.structure.slices_h:
                for s in zip(l[0], l[1]):
                    ax.plot(s[1], s[0], '.')
            ax.axis("off")
            name = "Map with slices"
            ax.set_title(name)
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["Partial Scores"]:
            co = len(self.structure.part_score)
            if co > 1:
                if co > 2:
                    div = he.proper_divs2(int(co))
                    if len(div) is 1:
                        co = co + 1
                        div = he.proper_divs2(int(co))
                    div.add(int(co))
                    div = list(div)
                    div.sort()
                    if len(div) % 2 == 0:
                        nrows = div[int(len(div) / 2) - 1]
                        ncols = div[int(len(div) / 2)]
                    else:
                        nrows = div[int(len(div) / 2)]
                        ncols = div[int(len(div) / 2)]
                else:
                    nrows = 1
                    ncols = int(co)
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
                i = 0
                for p in self.structure.part_score:
                    ax[int(i / ncols)][int(i % ncols)].imshow(p)
                    ax[int(i / ncols)][int(i % ncols)].axis('off')
                    i = i + 1
                name = "Partial Scores"
                fig.canvas.set_window_title(name)
                plt.show()
            elif co == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                ax.imshow(self.structure.part_score[0])
                ax.axis('off')
                name = "Partial Scores"
                fig.canvas.set_window_title(name)
                plt.show()

        if visualisation_flags["Partial Reconstructs"]:
            co = len(self.structure.part_reconstruction)
            if co > 1:
                if co > 2:
                    div = he.proper_divs2(int(co))
                    if len(div) is 1:
                        co = co + 1
                        div = he.proper_divs2(int(co))
                    div.add(int(co))
                    div = list(div)
                    div.sort()
                    if len(div) % 2 == 0:
                        nrows = div[int(len(div) / 2) - 1]
                        ncols = div[int(len(div) / 2)]
                    else:
                        nrows = div[int(len(div) / 2)]
                        ncols = div[int(len(div) / 2)]
                else:
                    nrows = 1
                    ncols = int(co)
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
                i = 0
                for p in self.structure.part_reconstruction:
                    ax[int(i / ncols)][int(i % ncols)].imshow(p)
                    ax[int(i / ncols)][int(i % ncols)].axis('off')
                    i = i + 1
                name = "Partial Reconstruct"
                fig.canvas.set_window_title(name)
                plt.show()
            elif co == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                ax.imshow(self.structure.part_reconstruction[0])
                ax.axis('off')
                name = "Partial Reconstruct"
                fig.canvas.set_window_title(name)
                plt.show()

        if visualisation_flags["Wall lines from mbb"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True)

            # cmap = plt.cm.get_cmap("tab10")
            # cmap.set_under("black")
            # cmap.set_over("yellow")
            #
            # ax.imshow(self.structure.labeled_map, cmap=cmap, vmin=1)
            ax = self.__show_patches(ax)
            ax = self.__show_long_segments(ax, self.structure.segments_h, self.structure.segments_h_mbb_lines)
            ax = self.__show_long_segments(ax, self.structure.segments_v, self.structure.segments_v_mbb_lines)
            # for local_segments, local_mbb_lines in zip(self.structure.segments_h, self.structure.segments_h_mbb_lines):
            #     for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
            #         ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
            #         if l_segment.mbb_area > 10:
            #             ax.plot([l_mbb_lines["Y1"], l_mbb_lines["Y2"]], [l_mbb_lines["X1"], l_mbb_lines["X2"]], 'g')
            # for local_segments, local_mbb_lines in zip(self.structure.segments_v, self.structure.segments_v_mbb_lines):
            #     for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
            #         ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
            #         if l_segment.mbb_area > 10:
            #             ax.plot([l_mbb_lines["Y1"], l_mbb_lines["Y2"]], [l_mbb_lines["X1"], l_mbb_lines["X2"]], 'g')
            ax.set_xlim(0, self.structure.binary_map.shape[1])
            ax.set_ylim(self.structure.binary_map.shape[0], 0)
            ax.axis("off")
            name = "Wall lines from mbb"
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["Labels and Raw map"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True)
            ax = self.__show_patches(ax)
            # cmap = plt.cm.get_cmap("tab10")
            # cmap.set_under("black")
            # cmap.set_over("yellow")
            #
            # ax.imshow(self.structure.labeled_map, cmap=cmap, vmin=1)
            ax.imshow(self.structure.binary_map, cmap="gray", alpha=0.5)
            for local_segments, local_mbb_lines in zip(self.structure.segments_h, self.structure.segments_h_mbb_lines):
                for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
                    ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
            for local_segments, local_mbb_lines in zip(self.structure.segments_v, self.structure.segments_v_mbb_lines):
                for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
                    ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
            ax.set_xlim(0, self.structure.binary_map.shape[1])
            ax.set_ylim(self.structure.binary_map.shape[0], 0)
            ax.axis("off")
            name = "Labels and Raw map"
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["Raw line segments"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True)
            ax.imshow(self.structure.binary_map, cmap="gray")
            for segment in self.structure.slice_v_lines:
                ax.plot([segment[1], segment[3]], [segment[0], segment[2]])

            for segment in self.structure.slice_h_lines:
                ax.plot([segment[1], segment[3]], [segment[0], segment[2]])

            name = "Raw line segments"
            fig.canvas.set_window_title(name)

            plt.show()

        if visualisation_flags["Short wall lines over original map"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True)
            ax.imshow(self.structure.binary_map, cmap="gray")
            # for local_segments, local_mbb_lines in zip(self.structure.segments_h, self.structure.segments_h_mbb_lines):
            #     for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
            #         ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
            #         wall = he.cetral_line(l_segment.minimal_bounding_box)
            #         ax.plot([wall[0].y, wall[1].y], [wall[0].x, wall[1].x], 'c')
            #         # if l_segment.mbb_area > 10:
            #         # ax.plot([l_mbb_lines["Y1"], l_mbb_lines["Y2"]], [l_mbb_lines["X1"], l_mbb_lines["X2"]], 'g')
            ax = self.__show_short_segments(ax, self.structure.segments_h, self.structure.segments_h_mbb_lines)
            ax = self.__show_short_segments(ax, self.structure.segments_v, self.structure.segments_v_mbb_lines)
            # for local_segments, local_mbb_lines in zip(self.structure.segments_v, self.structure.segments_v_mbb_lines):
            #     for l_segment, l_mbb_lines in zip(local_segments, local_mbb_lines):
            #         ax.plot(l_segment.minimal_bounding_box[:, 1], l_segment.minimal_bounding_box[:, 0], 'r')
            #         wall = he.cetral_line(l_segment.minimal_bounding_box)
            #         ax.plot([wall[0].y, wall[1].y], [wall[0].x, wall[1].x], 'c')
            #         # if l_segment.mbb_area > 10:
            #         # ax.plot([l_mbb_lines["Y1"], l_mbb_lines["Y2"]], [l_mbb_lines["X1"], l_mbb_lines["X2"]], 'g')
            ax.set_xlim(0, self.structure.binary_map.shape[1])
            ax.set_ylim(self.structure.binary_map.shape[0], 0)
            ax.axis("off")
            name = "Short wall lines over original map"
            fig.canvas.set_window_title(name)
            plt.show()

        if visualisation_flags["evaluation output"]:
            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax[0, 0].imshow(self.structure.binary_map, cmap="gray")
            ax[0, 0].axis("off")
            ax[0, 0].set_title("input padded map")

            # grounded_rose = self.structure.pol_h - min(self.structure.pol_h)
            # streached_rose = grounded_rose / max(grounded_rose)
            # signal_av = stat.mean(streached_rose)
            # peak_av = stat.mean(streached_rose[self.structure.peak_indices])
            #
            # ax[0,1].plot(self.structure.angles, streached_rose)
            # ax[0,1].plot(self.structure.angles[self.structure.peak_indices],
            #            streached_rose[self.structure.peak_indices], 'r+')
            # for p in self.structure.comp:
            #     ax[0,1].scatter(self.structure.angles[p], streached_rose[p], marker='^', s=120)
            #
            # ax[0,1].plot([self.structure.angles[0], self.structure.angles[-1]], [signal_av, signal_av])
            # ax[0,1].plot([self.structure.angles[0], self.structure.angles[-1]], [peak_av, peak_av])
            #
            # ax[0,1].set_ylabel("Normalised orientation score")
            # ax[0,1].set_ylim(0.0, 1.0)
            # ax[0,1].set_xlabel("Orientation [rad]")
            # ax[0,1].set_title("Unfolded FFT Spectrum")

            x = np.arange(np.min(self.structure.pixel_quality_histogram["edges"]),
                          np.max(self.structure.pixel_quality_histogram["edges"]),
                          (np.max(self.structure.pixel_quality_histogram["edges"]) - np.min(
                              self.structure.pixel_quality_histogram["edges"])) / 1000)

            if self.structure.pixel_quality_gmm["means"][0] < self.structure.pixel_quality_gmm["means"][1]:
                y_b = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][0],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][0])) * \
                      self.structure.pixel_quality_gmm["weights"][0]
                y_g = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][1],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][1])) * \
                      self.structure.pixel_quality_gmm["weights"][1]
            else:
                y_g = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][0],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][0])) * \
                      self.structure.pixel_quality_gmm["weights"][0]
                y_b = stats.norm.pdf(x, self.structure.pixel_quality_gmm["means"][1],
                                     math.sqrt(self.structure.pixel_quality_gmm["covariances"][1])) * \
                      self.structure.pixel_quality_gmm["weights"][1]

            ax[0, 1].bar(self.structure.pixel_quality_histogram["centers"],
                         self.structure.pixel_quality_histogram["bins"],
                         width=self.structure.pixel_quality_histogram["width"])
            ax[0, 1].plot(x, y_b, 'r')
            ax[0, 1].plot(x, y_g, 'g')
            # ax.axvline(x=self.structure.cluster_quality_threshold, color='y')
            ax[0, 1].axvline(x=self.structure.quality_threshold, color='y')
            ax[0, 1].set_title("Score trehsold estiamtion")

            ax[1, 0].imshow(np.abs(self.structure.map_scored), cmap="nipy_spectral")
            ax[1, 0].axis("off")
            ax[1, 0].set_title("Scored map")

            ax[1, 1] = self.__show_patches(ax[1, 1])
            ax[1, 1].set_title("Simple Filtered Map ({:.2f})".format(self.structure.quality_threshold))
            ax[1, 1].axis("off")

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(self.structure.analysed_map)

            plt.show()
        logging.debug("Visualisation generated in : %.2f s", time.time() - t)
