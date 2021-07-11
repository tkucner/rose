import numpy as np
import shapely.affinity as af


class StructureExtraction:
    def __init__(self, fft_filtered_map):
        self.fft_map = fft_filtered_map
        self.dominant_lines = []
        self.__transform_dominant_lines()

    def __transform_dominant_lines(self):
        for line in self.fft_map.lines:
            self.dominant_lines.append(af.rotate(line, np.pi, use_radians=True))

    def scan_for_walls(self):
        pass
