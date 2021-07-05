import itertools
import logging
import math
import time

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import shapely.affinity as af
import shapely.geometry as sg
from scipy import ndimage
from scipy.ndimage.interpolation import geometric_transform
from skimage.filters import threshold_yen
from sklearn import mixture

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    Specify the spline interpolation order.
    High orders may be slow for large images.
    :param img: input image
    :type img: ndarray
    :param order: spline interpolation order
    :type order: int
    :return: returns the converted image as well as vector of distances and angles
    :rtype: ndarray, (ndarray,ndarray)
    """

    # max_radius is the length of the diagonal
    # from a corner to the mid-point of img.
    max_radius = 0.5 * np.linalg.norm(img.shape)

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2 * np.pi * coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        # radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5 * img.shape[0] - radius * np.sin(theta)
        j = radius * np.cos(theta) + 0.5 * img.shape[1]
        return i, j

    polar = geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0, 1, img.shape[0])
    angs = np.linspace(0, 2 * np.pi, img.shape[1])

    return polar, (rads, angs)


def ang_dist(a, b):
    phi = np.abs(a - b) % (2 * np.pi)
    dist = (2 * np.pi - phi) if phi > np.pi else phi
    return dist


def get_mask(orientations, size, factor):
    intersections = []
    cells = []
    lines = []
    lines_vis = []
    base_line = sg.LineString([(0, 0), (0, 2 * size)])

    for orientation in orientations:
        line = af.translate(af.rotate(base_line, orientation, use_radians=True), size / 2, -size / 2)
        lines.append(line)
        lines_vis.append([line.coords[0][0], line.coords[0][1], line.coords[1][0], line.coords[1][1]])
        for v in range(0, size + 1):
            inter = line.intersection(sg.asLineString([(0, v), (size, v)]))
            if not inter.is_empty:
                intersections.append(inter)
                cells.append([int(inter.coords[0][0]), int(inter.coords[0][1])])
                cells.append([int(inter.coords[0][0]), int(inter.coords[0][1]) - 1])
        for h in range(0, size + 1):
            inter = line.intersection(sg.asLineString([(h, 0), (h, size)]))
            if not inter.is_empty:
                intersections.append(inter)
                cells.append([int(inter.coords[0][0]), int(inter.coords[0][1])])
                cells.append([int(inter.coords[0][0]) - 1, int(inter.coords[0][1])])

    cells.sort()
    cells = list(num for num, _ in itertools.groupby(cells))

    mask = np.zeros((size, size))
    for c in cells:
        if 0 <= c[0] < size and 0 <= c[1] < size:
            mask[c[1], c[0]] = 1
    for _ in range(factor):
        mask = ndimage.binary_dilation(mask).astype(mask.dtype)

    return np.flipud(mask), lines_vis


def get_gmm_threshold(values):
    clf = mixture.GaussianMixture(n_components=2)
    clf.fit(values.ravel().reshape(-1, 1))
    gmm = {"means": clf.means_, "weights": clf.weights_, "covariances": clf.covariances_}

    # bins, edges = np.histogram(values.ravel(), density=True)
    # x = np.arange(min(edges), max(edges), (max(edges) - min(edges)) / 1000)
    x = np.arange(min(values.ravel()), max(values.ravel()), (max(values.ravel()) - min(values.ravel())) / 1000)

    y1 = stats.norm.pdf(x, gmm["means"][0], math.sqrt(gmm["covariances"][0])) * gmm["weights"][0]
    y2 = stats.norm.pdf(x, gmm["means"][1], math.sqrt(gmm["covariances"][1])) * gmm["weights"][1]

    if gmm["means"][0] < gmm["means"][1]:
        y_b = y1
        y_g = y2
    else:
        y_g = y1
        y_b = y2
    ind = np.argmax(y_g > y_b)
    return x[ind], gmm


def binaryze_map(grid_map):
    thresh = threshold_yen(grid_map)
    binary_map = grid_map <= thresh
    return binary_map


def pad_map(grid_map):
    # check if need padding
    temp_map = np.zeros((grid_map.shape[0] + (grid_map.shape[0] % 2), grid_map.shape[1] + (grid_map.shape[1] % 2)))
    if not temp_map.shape[0] == temp_map.shape[1]:
        temp_map = np.zeros((np.max(temp_map.shape), np.max(temp_map.shape)), dtype=bool)
    temp_map[:grid_map.shape[0], :grid_map.shape[1]] = grid_map
    return temp_map


def get_histogram(values):
    """
        Args:
            values:
        """
    bins, edges = np.histogram(values.ravel(), density=True)
    histogram = {"bins": bins, "edges": edges,
                 "centers": [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])],
                 "width": [(a - b) for a, b in zip(edges[:-1], edges[1:])]}
    return histogram


class FFTFiltering:
    # def __init__(self, grid_map, ang_tr=0.01, peak_height=0.5, par=1, smooth=False, sigma=3):
    def __init__(self, grid_map, **kwargs):

        self.lines = []
        self.quality_threshold = []
        self.ang_tr = kwargs["angle_threshold"]
        self.peak_height = kwargs["peak_height"]
        self.par = kwargs["window_width"]
        self.smooth = kwargs["smooth_histogram"]
        self.sigma = kwargs["sigma"]
        self.grid_map = []
        self.binary_map = None
        self.analysed_map = None

        self.pixel_quality_histogram = []
        self.pixel_quality_gmm = []

        self.norm_ft_image = None
        self.pol = []
        self.angles = []
        self.pol_h = []
        self.peak_indices = []
        self.rads = []
        self.comp = []
        self.mask_ft_image = []
        self.map_scored_good = []

        self.ft_image = []
        self.dom_dirs = []

        self.__load_map(grid_map)

    def __load_map(self, grid_map):
        """
        Function reads input map and adjust it to requirements of the code. That is pads it to be square and converts it
        to 2D map.
        :param grid_map: input map
        :type grid_map: ndarray
        """
        ti = time.time()  # logging time

        if len(grid_map.shape) == 3:  # if map is multilayered keep only one
            grid_map = grid_map[:, :, 1]

        # binarize the mpa
        self.binary_map = binaryze_map(grid_map)

        # thresh = threshold_yen(grid_map)
        # self.binary_map = grid_map <= thresh
        # self.binary_map = self.binary_map

        # # pad map to have even number of pixels in each dimension (DFT behaves better)
        # if self.binary_map.shape[0] % 2 != 0:
        #     t = np.zeros((self.binary_map.shape[0] + 1, self.binary_map.shape[1]), dtype=bool)
        #     t[:-1, :] = self.binary_map
        #     self.binary_map = t
        # if self.binary_map.shape[1] % 2 != 0:
        #     t = np.zeros((self.binary_map.shape[0], self.binary_map.shape[1] + 1), dtype=bool)
        #     t[:, :-1] = self.binary_map
        #     self.binary_map = t
        #
        # # pad with zeros to square (easier to compute directions)
        # square_map = np.zeros((np.max(self.binary_map.shape), np.max(self.binary_map.shape)), dtype=bool)
        # square_map[:self.binary_map.shape[0], :self.binary_map.shape[1]] = self.binary_map

        self.binary_map = pad_map(self.binary_map)
        self.analysed_map = self.binary_map.copy()

        # logging
        logging.debug("Map loaded in %.2f s", time.time() - ti)
        logging.info("Map Shape: %d x %d", self.binary_map.shape[0], self.binary_map.shape[1])

    def __compute_fft(self):
        """
        Computes FFT image of self.binary_map and normalised (scaled 0-255) version of it.
        """
        ti = time.time()  # logging time
        self.ft_image = np.fft.fftshift(np.fft.fft2(self.binary_map * 1))
        self.norm_ft_image = (np.abs(self.ft_image) / np.max(np.abs(self.ft_image))) * 255.0
        self.norm_ft_image = self.norm_ft_image.astype(int)

        # logging
        logging.debug("FFT computed in: %.2f s", time.time() - ti)

    def __get_dominant_directions(self):
        """
        Detects peaks in unfolded FFT spectrum.
        """
        ti = time.time()  # logging time

        # unfold spectrum
        self.pol, (self.rads, self.angles) = topolar(self.norm_ft_image, order=3)

        # concatenate three frequency images to prevent peak distortion on the fringes of the image
        pol_l = self.pol.shape[1]
        self.pol = np.concatenate((self.pol, self.pol[:, 1:], self.pol[:, 1:]), axis=1)
        self.angles = np.concatenate(
            (self.angles, self.angles[1:] + np.max(self.angles), self.angles[1:] + np.max(self.angles[1:] +
                                                                                          np.max(self.angles))), axis=0)
        self.pol_h = np.array([sum(x) for x in zip(*self.pol)])

        # smooth the hisotgram
        if self.smooth:
            self.pol_h = ndimage.gaussian_filter1d(self.pol_h, self.sigma)

        # perform peak detection
        self.peak_indices, _ = signal.find_peaks(self.pol_h,
                                                 prominence=(np.max(self.pol_h) - np.min(
                                                     self.pol_h)) * self.peak_height)

        # remove the padding from the frequnecy spectrum
        self.pol = self.pol[:, 0:pol_l]
        self.angles = self.angles[0:pol_l]
        self.pol_h = self.pol_h[0:pol_l]
        self.peak_indices = self.peak_indices[np.logical_and(self.peak_indices >= pol_l - 1,
                                                             self.peak_indices < 2 * pol_l - 2)] - pol_l + 1

        # FFT spectrum is symmetric. Thus there will be always two peaks corresponding to one single wall.
        combinations = itertools.combinations(self.peak_indices, 2)
        for c in combinations:
            a = self.angles[c[0]]
            b = self.angles[c[1]]
            if math.isclose(np.pi - ang_dist(a, b), 0, abs_tol=self.ang_tr):
                self.comp.append([c[0], c[1]])
                logging.info("Found direction %.2f, %.2f", self.angles[c[0]] * 180.0 / np.pi,
                             self.angles[c[1]] * 180.0 / np.pi)
                self.dom_dirs.append([self.angles[c[0]], self.angles[c[1]]])
        logging.info("Number of directions: %d", len(self.comp))
        logging.debug("Directions computed in : %.2f s", time.time() - ti)

    def process_map(self):
        """

        """
        self.__compute_fft()
        self.__get_dominant_directions()

        ti = time.time()
        if not self.comp:
            pass
        else:
            angles = []
            for p in self.comp:
                angles.append((self.angles[p[0]] + self.angles[p[1]] - np.pi) / 2)
            mask_all, self.lines = get_mask(angles, self.binary_map.shape[0], self.par)
            self.mask_ft_image = self.ft_image * mask_all
            mask_iftimage = np.fft.ifft2(self.mask_ft_image)
            self.map_scored_good = np.abs(mask_iftimage) * (self.binary_map * 1)

        logging.debug("Map filtered in: %.2f s", time.time() - ti)

    def simple_filter_map(self, tr):
        """
        Args:
            tr:
        """
        ti = time.time()
        l_map = np.array(np.abs(self.map_scored_good) / np.max(np.abs(self.map_scored_good)))
        self.quality_threshold = tr
        self.analysed_map = self.binary_map.copy()
        self.analysed_map[l_map < self.quality_threshold] = 0.0
        logging.debug("Map filtered simple in : %.2f s", time.time() - ti)

    def histogram_filtering(self):
        ti = time.time()
        pixels = np.abs(self.map_scored_good[self.binary_map > 0])

        self.quality_threshold, self.pixel_quality_gmm = get_gmm_threshold(pixels)
        self.pixel_quality_histogram = get_histogram(pixels)

        self.analysed_map = self.binary_map.copy()
        self.analysed_map[np.abs(self.map_scored_good) < self.quality_threshold] = 0.0
        logging.debug("Map filtered with histogram in : %.2f s", time.time() - ti)
