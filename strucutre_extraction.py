import itertools
import time

import numpy as np
import shapely.affinity as af
import shapely.geometry as sg


def compute_corner(a, corner):
    return [a[0] + corner[0], a[1] + corner[1]]


def get_cell_bounding_box(cells):
    corners = np.array([[0.5, 0.5],
                        [-0.5, 0.5],
                        [-0.5, -0.5],
                        [0.5, -0.5]])
    temp_cell_corners = np.empty((0, 2), float)
    for corner in corners:
        new = np.apply_along_axis(compute_corner, 1, cells, corner=corner)
        temp_cell_corners = np.append(temp_cell_corners, np.array(new), axis=0)
    temp_multipoint = sg.MultiPoint(temp_cell_corners)
    return temp_multipoint.minimum_rotated_rectangle


def flood_fill(cells):
    labels = np.zeros(len(cells), dtype=int)
    l_cells = []
    for c in cells:
        l_cells.append((c[0], c[1]))
    cells_set = set(l_cells)
    label = 0
    for i, c in enumerate(cells):
        if labels[i] == 0:
            label = label + 1
            labels[i] = label

        toFill = set()

        toFill.add((c[0] - 1, c[1]))
        toFill.add((c[0] + 1, c[1]))

        toFill.add((c[0], c[1] - 1))
        toFill.add((c[0], c[1] + 1))

        toFill.add((c[0] - 1, c[1] + 1))
        toFill.add((c[0] - 1, c[1] - 1))

        toFill.add((c[0] + 1, c[1] - 1))
        toFill.add((c[0] + 1, c[1] + 1))

        adjacent = cells_set.intersection(toFill)

        for a in adjacent:
            if not labels[l_cells.index(a)] == 0 and not labels[l_cells.index(a)] == labels[i]:
                old_label = labels[l_cells.index(a)]
                for key, value in enumerate(labels):
                    if value == old_label:
                        labels[key] = labels[i]

            labels[l_cells.index(a)] = labels[i]

    unique_labels = list(set(labels))
    chunks = []

    for l in unique_labels:
        chunks.append(np.array(cells)[labels == l])

    return chunks


def central_line_approximation(cells):
    x = [c[0] for c in cells]
    y = [c[1] for c in cells]
    if max(x) - min(x) > max(y) - min(y):
        beginning = (x[x.index(min(x))], y[x.index(min(x))])
        end = (x[x.index(max(x))], y[x.index(max(x))])
    else:
        beginning = (x[y.index(min(y))], y[y.index(min(y))])
        end = (x[y.index(max(y))], y[y.index(max(y))])

    return beginning, end


class StructureExtraction:
    def __init__(self, fft_filtered_map):
        self.extracted_walls = []
        self.fft_map = fft_filtered_map
        self.dominant_lines = []
        self.wall_lines = []
        self.all_wall_cell_chunks = []
        self.__transform_dominant_lines()

    def __transform_dominant_lines(self):
        for line in self.fft_map.lines:
            self.dominant_lines.append(af.rotate(line, np.pi / 2., use_radians=True))

    def __directional_flood_filling(self, tr):
        for l in self.dominant_lines:
            print(l)
            start = time.time()
            all_pixels = []
            for wall_line in self.wall_lines:
                if l == wall_line["dominant_line"]:
                    for wall_chunk in wall_line["wall_cells_chunks"]:
                        if len(wall_chunk) > tr:
                            all_pixels.extend(wall_chunk)
            full_wall_chunks = flood_fill(all_pixels)

            processed_wall_chunk = []
            for fwc in full_wall_chunks:
                processed_wall_chunk.append({"cells": fwc, "minimal_bounding_box": get_cell_bounding_box(fwc)})
            self.extracted_walls.append(
                {"dominant_line": l, "cells": all_pixels, "wall_chunks": processed_wall_chunk})
            end = time.time()
            print("Getting full wall chunks took {}".format(end - start))

    def scan_for_walls(self):
        wall_cells = np.where(self.fft_map.binary_map)
        for line in self.dominant_lines:  # TODO parallelization
            used_cells = []
            start = time.time()
            for cell in zip(wall_cells[0], wall_cells[1]):
                if list(cell) not in used_cells:
                    l = sg.LineString([(line.coords[0][1], line.coords[0][0]), (line.coords[1][1], line.coords[1][0])])
                    wall_line = af.translate(l, xoff=cell[1] + 0.5 - l.centroid.x, yoff=cell[0] + 0.5 - l.centroid.y)
                    all_cells = []
                    for v in range(0, self.fft_map.binary_map.shape[0] + 1):
                        inter = wall_line.intersection(sg.asLineString([(0, v), (self.fft_map.binary_map.shape[0], v)]))
                        if not inter.is_empty:
                            all_cells.append([int(inter.coords[0][1]), int(inter.coords[0][0])])
                    for h in range(0, self.fft_map.binary_map.shape[1] + 1):
                        inter = wall_line.intersection(sg.asLineString([(h, 0), (h, self.fft_map.binary_map.shape[1])]))
                        if not inter.is_empty:
                            all_cells.append([int(inter.coords[0][1]), int(inter.coords[0][0])])
                    all_cells.sort()
                    all_cells = list(num for num, _ in itertools.groupby(all_cells))
                    local_wall_cells = []
                    for c in all_cells:
                        if 0 <= c[0] < self.fft_map.binary_map.shape[0] and 0 <= c[1] < self.fft_map.binary_map.shape[
                            1]:
                            if self.fft_map.binary_map[c[0], c[1]]:
                                local_wall_cells.append(c)
                                used_cells.append(c)

                    result_dict = {"cell": cell, "wall_line": wall_line, "all_cells": all_cells,
                                   "wall_cells": local_wall_cells, "dominant_line": line}
                    result_dict["wall_cells_chunks"] = flood_fill(result_dict["wall_cells"])
                    wall_line_chunks = []
                    for cells in result_dict["wall_cells_chunks"]:
                        if len(cells) > 1:
                            wall_line_chunks.append(central_line_approximation(cells))
                        else:
                            wall_line_chunks.append((([], []), ([], [])))
                    result_dict["wall_line_chunks"] = wall_line_chunks
                    self.all_wall_cell_chunks.extend(result_dict["wall_cells_chunks"])
                    self.wall_lines.append(result_dict)
            end = time.time()
            print("Line wall scanning took: {}".format(end - start))

        self.__directional_flood_filling(15)
