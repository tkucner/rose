import logging
import math
import os
import time

import matplotlib.collections as clt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import asarray, concatenate, ones

import helpers

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# from https://pypi.org/project/descartes/
class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface
    def __init__(self, context):
        if isinstance(context, dict):
            self.context = context['coordinates']
        else:
            self.context = context

    @property
    def exterior(self):
        return (getattr(self.context, 'exterior', None)
                or self.context[0])

    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context[1:]
        return value


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    if hasattr(polygon, 'geom_type'):  # Shapely
        ptype = polygon.geom_type
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    else:  # GeoJSON
        polygon = getattr(polygon, '__geo_interface__', polygon)
        ptype = polygon["type"]
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon['coordinates']]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    vertices = concatenate([
        concatenate([asarray(t.exterior)[:, :2]] +
                    [asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])

    codes = concatenate([
        concatenate([coding(t.exterior)] +
                    [coding(r) for r in t.interiors]) for t in polygon])
    r_vertices = [[row[1], row[0]] for row in vertices]

    return Path(r_vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPath(polygon), **kwargs)


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
            plt.xlim(0, self.fft_filtered.grid_map.shape[1])
            plt.ylim(0, self.fft_filtered.grid_map.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Binary map"]:
            name = "Binary map"
            plt.figure()
            plt.imshow(self.fft_filtered.binary_map, cmap="gray")
            plt.xlim(0, self.fft_filtered.binary_map.shape[1])
            plt.ylim(0, self.fft_filtered.binary_map.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Frequency image"]:
            name = "Frequency image"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.frequency_image), cmap="nipy_spectral")
            plt.xlim(0, self.fft_filtered.frequency_image.shape[1])
            plt.ylim(0, self.fft_filtered.frequency_image.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Frequency image with filter"]:
            name = "Frequency image with filter"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.frequency_image), cmap="nipy_spectral")
            plt.imshow(np.logical_not(self.fft_filtered.filter) * 1, cmap='gray', alpha=0.25)
            plt.xlim(0, self.fft_filtered.frequency_image.shape[1])
            plt.ylim(0, self.fft_filtered.frequency_image.shape[0])
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
            plt.xlim(0, self.fft_filtered.filter.shape[1])
            plt.ylim(0, self.fft_filtered.filter.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Filtered frequency image"]:
            name = "Filtered frequency image"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.filtered_frequency_image), cmap="nipy_spectral")
            plt.xlim(0, self.fft_filtered.filtered_frequency_image.shape[1])
            plt.ylim(0, self.fft_filtered.filtered_frequency_image.shape[0])
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
            plt.xlim(0, self.fft_filtered.reconstructed_map.shape[1])
            plt.ylim(0, self.fft_filtered.reconstructed_map.shape[0])
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Scored map"]:
            name = "Scored map"
            plt.figure()
            plt.imshow(np.abs(self.fft_filtered.map_scored), cmap="nipy_spectral")
            plt.xlim(0, self.fft_filtered.map_scored.shape[1])
            plt.ylim(0, self.fft_filtered.map_scored.shape[0])
            plt.colorbar()
            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Filtered map overlay"]:
            name = "Filtered map overlay (" + str(self.fft_filtered.quality_threshold) + ")"
            plt.figure()
            plt.imshow(np.flipud(np.abs(self.fft_filtered.binary_map)), cmap="gray")
            plt.imshow(np.flipud(np.logical_not(self.fft_filtered.analysed_map)) * 1, cmap='gray', alpha=0.75)

            plt.title(name)
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Filtered map final"]:
            name = "Filtered map final (" + str(self.fft_filtered.quality_threshold) + ")"
            plt.figure()
            plt.imshow(np.flipud(np.logical_not(self.fft_filtered.analysed_map)) * 1, cmap='gray')

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
            start = time.time()
            name = "Map with directions"
            plt.figure()
            plt.imshow(self.structured_map.fft_map.binary_map, cmap="gray")
            for l in self.structured_map.dominant_lines:
                plt.plot([l.coords[0][1], l.coords[1][1]], [l.coords[0][0], l.coords[1][0]])
            plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
            plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])
            plt.title(name)
            end = time.time()
            print("Ploting {} took {}".format(name, end - start))
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Map with long wall lines"]:
            start = time.time()
            name = "Map with long wall lines"
            plt.figure()
            plt.imshow(self.structured_map.fft_map.binary_map, cmap="gray")
            plot = True
            points = []
            crossess = []
            lines = []
            for l in self.structured_map.wall_lines:
                # lines = []
                if len(l["wall_cells"]) > 10:
                    plot = False
                    lines.append(((l["wall_line"].coords[1][0] - 0.5, l["wall_line"].coords[1][1] - 0.5),
                                  (l["wall_line"].coords[0][0] - 0.5, l["wall_line"].coords[0][1] - 0.5)))
                    crossess.extend(l["all_cells"])
                    points.extend(l["wall_cells"])

            plt.plot([row[1] for row in crossess], [row[0] for row in crossess], 'rx')
            plt.plot([row[1] for row in points], [row[0] for row in points], 'g.')
            ln_coll = clt.LineCollection(lines)
            plt.gca().add_collection(ln_coll)
            plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
            plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])
            plt.title(name)
            end = time.time()
            print("Plotting {} took {}".format(name, end - start))
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Map with short wall lines"]:
            start = time.time()
            name = "Map with short wall lines"
            plt.figure()
            plt.imshow(self.structured_map.fft_map.binary_map, cmap="gray")
            lines = []
            for wl in self.structured_map.wall_lines:
                for wlc in wl["wall_line_chunks"]:
                    if not wlc[0] == ([], []):
                        wlc_flip = ((wlc[0][1], wlc[0][0]), (wlc[1][1], wlc[1][0]))
                        lines.append(wlc_flip)

            ln_coll = clt.LineCollection(lines)
            plt.gca().add_collection(ln_coll)
            plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
            plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])

            plt.title(name)
            end = time.time()
            print("Plotting {} took {}".format(name, end - start))
            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)

        if self.flags["Separated wall directions"]:
            start = time.time()
            name = "Separated wall directions"

            for ew in self.structured_map.extracted_walls:
                local_map = np.zeros(self.structured_map.fft_map.binary_map.shape)
                plt.figure()
                for c in ew["cells"]:
                    local_map[c[0], c[1]] = 255
                plt.imshow(local_map, cmap="gray")
                plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
                plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])
                plt.title(name)

                if self.flags["Save path"] != "":
                    save_plot(self.save_dir, name)
            end = time.time()
            print("Plotting {} took {}".format(name, end - start))

        if self.flags["Wall pixels"]:
            plt.figure()
            start = time.time()
            name = "Wall pixels"
            plt.imshow(self.structured_map.fft_map.binary_map, cmap="gray")

            x = []
            y = []
            c = []
            for ew in self.structured_map.extracted_walls:
                counter = 0
                colors = iter([plt.cm.tab20(i) for i in range(20)])
                for wc in ew["wall_chunks"]:
                    x.extend([row[1] for row in wc["cells"]])
                    y.extend([row[0] for row in wc["cells"]])
                    if counter < 19:
                        counter = counter + 1
                    else:
                        colors = iter([plt.cm.tab20(i) for i in range(20)])
                        counter = 0
                    colour = next(colors)
                    c.extend([colour] * len([row[1] for row in wc["cells"]]))
            # plt.scatter(x, y, c=[next(colors)])
            plt.scatter(x, y, c=c)

            plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
            plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])
            plt.title(name)

            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)
            end = time.time()
            print("Plotting {} took {}".format(name, end - start))

        if self.flags["Wall bounding boxes"]:
            plt.figure()
            start = time.time()
            name = "Wall bounding boxes"
            plt.imshow(self.structured_map.fft_map.binary_map, cmap="gray")
            for ew in self.structured_map.extracted_walls:
                counter = 0
                colors = iter([plt.cm.tab20(i) for i in range(20)])
                for wc in ew["wall_chunks"]:
                    if counter < 19:
                        counter = counter + 1
                    else:
                        colors = iter([plt.cm.tab20(i) for i in range(20)])
                        counter = 0
                    colour = next(colors)

                    patch = PolygonPatch(wc["minimal_bounding_box"], facecolor=colour, edgecolor="w", alpha=0.5,
                                         zorder=2)
                    plt.gca().add_patch(patch)

            plt.xlim(0, self.structured_map.fft_map.binary_map.shape[1])
            plt.ylim(0, self.structured_map.fft_map.binary_map.shape[0])
            plt.title(name)

            if self.flags["Save path"] != "":
                save_plot(self.save_dir, name)
            end = time.time()
            print("Plotting {} took {}".format(name, end - start))

        if self.flags["Show plots"]:
            plt.show()
