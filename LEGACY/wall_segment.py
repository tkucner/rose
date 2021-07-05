import numpy as np
from shapely.geometry import MultiPoint


class WallSegment(MultiPoint):
    def __init__(self):
        self.id = -1
        self.cells = None

    @staticmethod
    def compute_corner(a, corner):
        """
        Args:
            a:
            corner:
        """
        return [a[0] + corner[0], a[1] + corner[1]]

    def add_cells(self, cells):
        """
        Args:
            cells:
        """
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
