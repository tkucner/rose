import itertools

import numpy as np
import shapely.affinity as af
import shapely.geometry as sg


def flood_fill(cells):
    labels = np.zeros(len(cells), dtype=int)
    l_cells = []
    for c in cells:
        l_cells.append((c[0], c[1]))
    cells_set = set(l_cells)
    label = 0
    equivalency_list = []
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

    # if max(unique_labels) == len(unique_labels):
    #     for l in range(1, int(max(unique_labels)) + 1):
    #         chunks.append(np.array(cells)[labels == l])
    # else:
    #     new_label=1
    #     for i in unique_labels:
    #         labels[labels==i]=new_label
    #         new_label=new_label+1
    #     unique_labels = list(set(labels))
    #     for l in range(1, int(max(unique_labels)) + 1):
    #         chunks.append(np.array(cells)[labels == l])
    for l in unique_labels:
        chunks.append(np.array(cells)[labels == l])

    return chunks


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
        # TODO over segementation in one direction! WHY????
        for l in self.dominant_lines:
            all_pixels = []

            for wall_line in self.wall_lines:
                if l == wall_line["dominant_line"]:
                    for wall_chunk in wall_line["wall_cells_chunks"]:
                        if len(wall_chunk) > tr:
                            all_pixels.extend(wall_chunk)

            full_wall_chunks = flood_fill(all_pixels)
            self.extracted_walls.append({"dominant_line": l, "cells": all_pixels, "wall_chunks": full_wall_chunks})

    def scan_for_walls(self):
        wall_cells = np.where(self.fft_map.binary_map)
        for line in self.dominant_lines:
            used_cells = []
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
                    self.all_wall_cell_chunks.extend(result_dict["wall_cells_chunks"])
                    self.wall_lines.append(result_dict)

        self.__directional_flood_filling(5)
