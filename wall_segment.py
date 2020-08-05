import numpy as np
from shapely.geometry import MultiPoint, LineString


class WallSegment(MultiPoint):
    def __init__(self):
        self.id = -1
        self.cells = None
        self.central_lines = {}
        self.square = False

    @staticmethod
    def compute_corner(a, corner):
        return [a[0] + corner[0], a[1] + corner[1]]

    def compute_central_lines(self):
        # get the sides of the mbb
        coords = self.minimum_rotated_rectangle.exterior.coords
        # compute the centers of the edges
        a = LineString(coords[0:2])
        b = LineString(coords[1:3])
        c = LineString(coords[2:4])
        d = LineString(coords[3:5])

        s1 = LineString([a.interpolate(0.5 * a.length), c.interpolate(0.5 * c.length)])
        s2 = LineString([b.interpolate(0.5 * b.length), d.interpolate(0.5 * d.length)])
        if s1.length > s2.length:
            self.central_lines['long'] = s1
            self.central_lines['short'] = s2
        elif s1.length < s2.length:
            self.central_lines['long'] = s2
            self.central_lines['short'] = s1
        elif s1.length == s2.length:
            self.central_lines['long'] = s1
            self.central_lines['short'] = s2
            self.square = True

    def add_cells(self, cells):
        corners = np.array([[0.5, 0.5],
                            [-0.5, 0.5],
                            [-0.5, -0.5],
                            [0.5, -0.5]])
        if self.cells is None:
            self.cells = cells

            temp_cell_corners = np.empty((0, 2), float)
            for corner in corners:
                new = np.apply_along_axis(WallSegment.compute_corner, 1, self.cells, corner=corner)
                temp_cell_corners = np.append(temp_cell_corners, np.array(new), axis=0)
            super().__init__(temp_cell_corners)
