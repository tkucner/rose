import itertools

import numpy as np
import shapely.affinity as af
import shapely.geometry as sg


class StructureExtraction:
    def __init__(self, fft_filtered_map):
        self.fft_map = fft_filtered_map
        self.dominant_lines = []
        self.wall_lines = []
        self.__transform_dominant_lines()

    def __transform_dominant_lines(self):
        for line in self.fft_map.lines:
            self.dominant_lines.append(af.rotate(line, np.pi, use_radians=True))

    def scan_for_walls(self):
        # TODO Some bug....

        wall_cells = np.where(self.fft_map.binary_map)
        for c in zip(wall_cells[0], wall_cells[1]):
            for l in self.dominant_lines:
                wall_line = af.translate(l, c[0] + 0.5 - l.centroid.x, c[1] + 0.5 - l.centroid.y)
                all_cells = []
                for v in range(0, self.fft_map.binary_map.shape[0] + 1):
                    inter = wall_line.intersection(sg.asLineString([(0, v), (self.fft_map.binary_map.shape[0], v)]))
                    if not inter.is_empty:
                        # intersections.append(inter)
                        all_cells.append([int(inter.coords[0][0]), int(inter.coords[0][1])])
                        all_cells.append([int(inter.coords[0][0]), int(inter.coords[0][1]) - 1])
                for h in range(0, self.fft_map.binary_map.shape[1] + 1):
                    inter = wall_line.intersection(sg.asLineString([(h, 0), (h, self.fft_map.binary_map.shape[1])]))
                    if not inter.is_empty:
                        # intersections.append(inter)
                        all_cells.append([int(inter.coords[0][0]), int(inter.coords[0][1])])
                        all_cells.append([int(inter.coords[0][0]) - 1, int(inter.coords[0][1])])
                all_cells.sort()
                all_cells = list(num for num, _ in itertools.groupby(all_cells))
                wall_cells = []
                for c in all_cells:
                    if 0 <= c[0] < self.fft_map.binary_map.shape[0] and 0 <= c[1] < self.fft_map.binary_map.shape[1]:
                        if self.fft_map.binary_map[c[0], c[1]]:
                            wall_cells.append(c)

                result_dict = {"wall_line": wall_line, "all_cells": all_cells, "wall_cells": wall_cells}
                self.wall_lines.append(result_dict)
