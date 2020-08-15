import logging
import math
import time

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import skimage.draw as sk_draw
from GridMapDecompose import segment_handling as sh
from scipy import ndimage
from skimage.filters import threshold_yen
from skimage.morphology import binary_dilation
from skimage.segmentation import flood_fill
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

import helpers as he
from wall_segment import WallSegment

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


class FFTStructureExtraction:
    def __init__(self, grid_map, ang_tr=0.1, amp_tr=0.8, peak_height=0.5, par=50, smooth=False, sigma=3):
        self.clustering_v_labels = []
        self.slice_v_lines = []
        self.slice_h_lines = []
        self.clustering_h_labels = []
        self.all_lines = []
        self.segments_h_mbb_lines = []
        self.segments_v_mbb_lines = []
        self.segments_h = []
        self.segments_v = []
        self.slices_h_dir = []
        self.slices_v_dir = []
        self.slices_v_ids = []
        self.slices_h_ids = []
        self.slices_v = []
        self.slices_h = []
        self.scored_hypothesis_v_cut = []
        self.scored_hypothesis_h_cut = []
        self.part_reconstruction = []
        self.lines = []
        self.lines_long_h = []
        self.lines_long_v = []
        self.cell_hypothesis_h = []
        self.cell_hypothesis_v = []
        self.lines_hypothesis_h = []
        self.lines_hypothesis_v = []
        self.scored_hypothesis_h = []
        self.scored_hypothesis_v = []
        self.d_row_h = []
        self.d_row_v = []
        self.part_mask = []
        self.part_score = []
        self.quality_threshold = []

        self.ang_tr = ang_tr  # rad
        self.amp_tr = amp_tr  # ratio
        self.peak_height = peak_height
        self.par = par
        self.smooth = smooth
        self.sigma = sigma
        self.grid_map = []
        self.binary_map = None
        self.analysed_map = None
        self.smooth = smooth

        self.pixel_quality_histogram = []
        self.pixel_quality_gmm = []
        # self.cluster_quality_threshold = []
        self.map_size = None
        self.line_parameters = []
        self.norm_ft_image = None
        self.pol = []
        self.angles = []
        self.pol_h = []
        self.peak_indices = []
        self.rads = []
        self.comp = []
        self.mask_ft_image = []
        self.mask_inv_ft_image = []
        self.map_scored_good = []
        self.map_scored_bad = []
        self.map_scored_diff = []
        self.map_split_good = []
        self.ft_image_split = []
        self.ft_image = []
        self.map_split_good_t = []
        self.dom_dirs = []

        self.__load_map(grid_map)

    # Static methods

    @staticmethod
    def __get_gmm_threshold(values):
        """
        Function generates a treshold assuming that the data is one dimensional and is buidl out of two normally distributed populations.
        The treshold is set as a value where two distributions has equal value.

        :param values: 1D vector of data
        :return: the value of the trehsold and the gaussian mixture model
        """
        clf = mixture.GaussianMixture(n_components=2)
        clf.fit(values.ravel().reshape(-1, 1))
        gmm = {"means": clf.means_, "weights": clf.weights_, "covariances": clf.covariances_}
        # v_range = (np.max(values) - np.min(values))

        bins, edges = np.histogram(values.ravel(), density=True)
        x = np.arange(min(edges), max(edges), (max(edges) - min(edges)) / 1000)

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

    @staticmethod
    def __get_histogram(values):
        """
        Wrapper function for numpy historgram,
        :param values: 1D vector of values
        :return: dict containing historgam values
        """
        bins, edges = np.histogram(values.ravel(), density=True)
        histogram = {"bins": bins, "edges": edges,
                     "centers": [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])],
                     "width": [(a - b) for a, b in zip(edges[:-1], edges[1:])]}
        return histogram

    @staticmethod
    def __generate_line_segments_per_direction(slices):
        """
        Function merging the slices a long lines to one list

        :param slices: 3D list of slices
        :return: list of slices
        """
        slices_lines = []
        for s in slices:
            slices_lines.append([s[0][0][0], s[1][0][0], s[0][0][-1], s[1][0][-1]])
        return slices_lines

    @staticmethod
    def __merge_walls(wall_segments, merge):
        """
        Function merge overlaping wall segsemnts.

        :param wall_segments: List of interacting wall segeemnts
        :param merge: array defining which segments to merge
        :return: lsit of mergeged walls, list of walls droped after merging
        """
        np_merge = np.array(merge)
        merge_list = list(zip(*np.where(np_merge == True)))
        merge_list = [set(x) for x in merge_list]
        merge_list = he.tuple_list_merger(merge_list)
        remove_list = []
        done_list = []
        for mi in merge_list:
            new_cells = []
            for m in mi:
                new_cells.extend(list(wall_segments[m].cells))
                remove_list.append(wall_segments[m])
            WS = WallSegment()
            WS.add_cells(np.array(new_cells))
            WS.compute_central_lines()
            done_list.append(WS)

        for w in wall_segments:
            if w not in remove_list:
                done_list.append(w)

        return done_list, remove_list

    @staticmethod
    def __process_wall_cluster(wall_segments, intersection_ratio_threshold):
        """
        Function checks if the walls are overlaping enough and label them to be merged.
        :param wall_segments: list of interacting wall segements
        :param intersection_ratio_threshold: proejction ration for merging
        :return: 2d array dentoing which segemnts interact, 2d array
        """
        interaction = [[False for x in range(len(wall_segments))] for y in range(len(wall_segments))]
        projections = [[None for x in range(len(wall_segments))] for y in range(len(wall_segments))]
        intersections = [[None for x in range(len(wall_segments))] for y in range(len(wall_segments))]
        intersections_ratios = [[None for x in range(len(wall_segments))] for y in range(len(wall_segments))]
        merge = [[False for x in range(len(wall_segments))] for y in range(len(wall_segments))]
        for ws1_id, ws1 in enumerate(wall_segments):
            for ws2_id, ws2 in enumerate(wall_segments):
                if not ws1_id == ws2_id:
                    interaction[ws1_id][ws2_id] = not ws1.minimum_rotated_rectangle.disjoint(
                        ws2.minimum_rotated_rectangle)
                    if interaction[ws1_id][ws2_id]:
                        s = he.orthogonal_projection(list(ws1.central_lines['short'].coords),
                                                     ws2.minimum_rotated_rectangle.bounds)
                        projections[ws1_id][ws2_id] = s
                        intersections[ws1_id][ws2_id] = s.difference(ws1.central_lines['short'])
                        intersections_ratios[ws1_id][ws2_id] = intersections[ws1_id][ws2_id].length / s.length
                        if intersections_ratios[ws1_id][ws2_id] < intersection_ratio_threshold:
                            merge[ws1_id][ws2_id] = True
        return interaction, projections, intersections, intersections_ratios, merge

    @staticmethod
    def __slice_wall(cc, rr, flag, l_slice_ids, cell_tr, new_row, vert, line_long, temp_row_full, temp_row_cut, s):
        """

        :param cc: column ids for the slice
        :param rr: row ids for the slice
        :param flag: flag if the cell is a wall or not
        :param l_slice_ids: line slice ids
        :param cell_tr: cell treshold
        :param new_row: flag is new row porceesed in direction
        :param vert: flag if line is vertical
        :param line_long: line long paremeters
        :param temp_row_full: full row of cells
        :param temp_row_cut: cuted row of cells
        :param s: offset
        :return:
        """
        # d_row_ret = []
        slices_ids = []
        slices = []
        cell_hypothesis = []
        lines_hypothesis = []
        kde_hypothesis = []
        kde_hypothesis_cut = []
        # slices_dir = []
        temp_slice = []

        cc_f = cc[flag]
        rr_f = rr[flag]
        cc_slices = []
        rr_slices = []

        for tslice in l_slice_ids:
            if len(tslice) > cell_tr:
                cc_s = []
                rr_s = []
                for i in tslice:
                    cc_s.append(cc_f[i])
                    rr_s.append(rr_f[i])
                cc_slices.append(cc_s)
                rr_slices.append(rr_s)

                temp_slice.append((cc_slices, rr_slices))

                slices_ids.append((cc_slices, rr_slices))

                slices.append((cc_slices, rr_slices))
                if new_row:

                    cell_hypothesis.append((cc[flag], rr[flag]))
                    if vert:
                        lines_hypothesis.append(
                            [line_long[0] + s, line_long[1], line_long[2] + s, line_long[3]])
                    elif not vert:
                        lines_hypothesis.append(
                            [line_long[0], line_long[1] + s, line_long[2], line_long[3] + s])
                    kde_hypothesis.append(temp_row_full)
                    kde_hypothesis_cut.append(temp_row_cut)
                    new_row = False

        return slices_ids, slices, cell_hypothesis, lines_hypothesis, kde_hypothesis, kde_hypothesis_cut, temp_slice, \
               new_row

    # Private methods

    def __get_cell_ids(self, vert, line_long, s):
        """
        Function computing cell ids along the lines it also computes the ids of occupied cells
        :param vert: flag is line is vertical
        :param line_long: long line crossing the map
        :param s: offset
        :return: list of cell ids as row anc coulumn ids, row and column flags, joint clags
        """
        if vert:
            rr, cc = sk_draw.line(int(round(line_long[0] + s)), int(round(line_long[3])),
                                  int(round(line_long[2] + s)),
                                  int(round(line_long[1])))
        elif not vert:
            rr, cc = sk_draw.line(int(round(line_long[0])), int(round(line_long[3] + s)),
                                  int(round(line_long[2])),
                                  int(round(line_long[1] + s)))
        rr_flag = (np.logical_or(rr < 0, rr >= self.analysed_map.shape[1]))
        cc_flag = (np.logical_or(cc < 0, cc >= self.analysed_map.shape[0]))
        flag = np.logical_not(np.logical_or(rr_flag, cc_flag))
        return rr, cc, rr_flag, cc_flag, flag

    def __load_map(self, grid_map):
        """
        Function loads maps form image file to the internal array
        :param grid_map: map in pgm format
        """
        ti = time.time()
        if len(grid_map.shape) == 3:
            grid_map = grid_map[:, :, 1]
        thresh = threshold_yen(grid_map)
        self.binary_map = grid_map <= thresh
        self.binary_map = self.binary_map
        if self.binary_map.shape[0] % 2 != 0:
            t = np.zeros((self.binary_map.shape[0] + 1, self.binary_map.shape[1]), dtype=bool)
            t[:-1, :] = self.binary_map
            self.binary_map = t
        if self.binary_map.shape[1] % 2 != 0:
            t = np.zeros((self.binary_map.shape[0], self.binary_map.shape[1] + 1), dtype=bool)
            t[:, :-1] = self.binary_map
            self.binary_map = t
        # pad with zeros to square
        square_map = np.zeros((np.max(self.binary_map.shape), np.max(self.binary_map.shape)), dtype=bool)
        square_map[:self.binary_map.shape[0], :self.binary_map.shape[1]] = self.binary_map
        self.binary_map = square_map
        self.map_size = square_map.shape[0]
        self.analysed_map = self.binary_map.copy()
        logging.debug("Map loaded in %.2f s", time.time() - ti)
        logging.info("Map Shape: %d x %d", self.binary_map.shape[0], self.binary_map.shape[1])

    def __compute_fft(self):
        """
        Functions computes fft of a map
        """
        ti = time.time()
        self.ft_image = np.fft.fftshift(np.fft.fft2(self.binary_map * 1))
        self.norm_ft_image = (np.abs(self.ft_image) / np.max(np.abs(self.ft_image))) * 255.0
        self.norm_ft_image = self.norm_ft_image.astype(int)

        logging.debug("FFT computed in: %.2f s", time.time() - ti)

    def __generate_mask(self, x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2, y_org):
        """

        :param x1_1:
        :param y1_1:
        :param x2_1:
        :param y2_1:
        :param x1_2:
        :param y1_2:
        :param x2_2:
        :param y2_2:
        :param y_org:
        :return:
        """
        mask_1 = np.zeros(self.norm_ft_image.shape, dtype=np.uint8)
        c_1 = np.array([y1_1, y2_1, self.norm_ft_image.shape[0], self.norm_ft_image.shape[0]])
        r_1 = np.array([x1_1, x2_1, self.norm_ft_image.shape[0], 0])
        if np.abs(y_org) > 3 * np.max(self.binary_map.shape):
            c_1 = np.array([y1_1, y2_1, self.norm_ft_image.shape[0], 0])
            r_1 = np.array([x1_1, x2_1, 0, 0])
        rr, cc = he.generate_mask(r_1, c_1, self.norm_ft_image.shape)
        mask_1[rr, cc] = 1
        mask_1 = np.flipud(mask_1)

        mask_2 = np.zeros(self.norm_ft_image.shape, dtype=np.uint8)
        c_2 = np.array([y1_2, y2_2, 0, 0])
        r_2 = np.array([x1_2, x2_2, self.norm_ft_image.shape[0], 0])
        if np.abs(y_org) > 3 * np.max(self.binary_map.shape):
            c_2 = np.array([y1_2, y2_2, self.norm_ft_image.shape[0], 0])
            r_2 = np.array([x1_2, x2_2, self.norm_ft_image.shape[0], self.norm_ft_image.shape[0]])
        rr, cc = he.generate_mask(r_2, c_2, self.norm_ft_image.shape)
        mask_2[rr, cc] = 1
        mask_2 = np.flipud(mask_2)

        mask_l = np.logical_and(mask_1, mask_2)
        return mask_l

    def __get_dominant_directions(self):
        ti = time.time()
        self.pol, (self.rads, self.angles) = he.topolar(self.norm_ft_image, order=3)
        pol_l = self.pol.shape[1]
        self.pol = np.concatenate((self.pol, self.pol[:, 1:], self.pol[:, 1:]), axis=1)
        self.angles = np.concatenate(
            (self.angles, self.angles[1:] + np.max(self.angles), self.angles[1:] + np.max(self.angles[1:] +
                                                                                          np.max(self.angles))), axis=0)

        if self.smooth:
            self.angles = ndimage.gaussian_filter1d(self.angles, self.sigma)
            self.pol = ndimage.gaussian_filter1d(self.pol, self.sigma)

        self.pol_h = np.array([sum(x) for x in zip(*self.pol)])

        self.peak_indices, _ = signal.find_peaks(self.pol_h,
                                                 prominence=(np.max(self.pol_h) - np.min(
                                                     self.pol_h)) * self.peak_height)

        self.pol = self.pol[:, 0:pol_l]
        self.angles = self.angles[0:pol_l]
        self.pol_h = self.pol_h[0:pol_l]
        self.peak_indices = self.peak_indices[np.logical_and(self.peak_indices >= pol_l - 1,
                                                             self.peak_indices < 2 * pol_l - 2)] - pol_l + 1

        pairs = list()
        for aind in self.peak_indices:
            for bind in self.peak_indices:
                a = self.angles[aind]
                b = self.angles[bind]
                if np.abs(np.pi - he.ang_dist(a, b)) < self.ang_tr:
                    pairs.append([aind, bind])

        if pairs:
            pairs = np.array(pairs)
            pairs = np.unique(np.sort(pairs), axis=0)

        amp = np.max(self.pol_h) - np.min(self.pol_h)
        self.comp = list()
        for p in pairs:
            a = self.pol_h[p[0]]
            b = self.pol_h[p[1]]
            if np.abs(a - b) / amp < self.amp_tr:
                self.comp.append(p)
        logging.debug("Directions computed in : %.2f s", time.time() - ti)
        logging.info("Number of directions: %d", len(self.comp))
        for p in self.comp:
            logging.info("Found direction %.2f, %.2f", self.angles[p[0]] * 180.0 / np.pi,
                         self.angles[p[1]] * 180.0 / np.pi)
            self.dom_dirs.append([self.angles[p[0]], self.angles[p[1]]])

    def __generate_initial_hypothesis_direction_with_kde(self, lines_long, max_len, bandwidth, cutoff_percent, cell_tr,
                                                         vert):
        d_row_ret = []
        slices_ids = []
        slices = []
        cell_hypothesis = []
        lines_hypothesis = []
        kde_hypothesis = []
        kde_hypothesis_cut = []
        slices_dir = []

        for line_long in lines_long:
            temp_slice = []
            for s in np.arange(-1 * max_len, max_len, 1):
                rr, cc, rr_flag, cc_flag, flag = self.__get_cell_ids(vert, line_long, s)
                new_row = True
                if np.sum(self.analysed_map[cc[flag], rr[flag]] * 1) > 1:
                    l_slice_ids, temp_row_full, temp_row_cut, d_row = self.__get_slices_along_the_line_kde(cc[flag],
                                                                                                           rr[flag],
                                                                                                           bandwidth,
                                                                                                           max_len,
                                                                                                           cutoff_percent)
                    d_row_ret.append(d_row)
                    l_slices_ids, l_slices, l_cell_hypothesis, l_lines_hypothesis, l_kde_hypothesis, l_kde_hypothesis_cut, l_temp_slice, new_row = self.__slice_wall(
                        cc, rr, flag, l_slice_ids, cell_tr, new_row, vert, line_long, temp_row_full, temp_row_cut, s)
                    slices_ids.extend(l_slices_ids)
                    slices.extend(l_slices)
                    cell_hypothesis.extend(l_cell_hypothesis)
                    lines_hypothesis.extend(l_lines_hypothesis)
                    kde_hypothesis.extend(l_kde_hypothesis)
                    kde_hypothesis_cut.extend(l_kde_hypothesis_cut)
                    temp_slice.extend(l_temp_slice)

            slices_dir.append(temp_slice)
        return d_row_ret, slices_ids, slices, cell_hypothesis, lines_hypothesis, \
               kde_hypothesis, kde_hypothesis_cut, slices_dir

    def __estimate_wall_thickness(self):
        max_len = 5000
        padding = 1
        thickness = np.array(self.__estimate_wall_thickness_in_direction(self.lines_long_v, max_len, padding, True) +
                             self.__estimate_wall_thickness_in_direction(self.lines_long_h, max_len, padding, False))
        thickness_threshold, thickness_gmm = self.__get_gmm_threshold(thickness)
        return thickness_threshold

    def __get_slices_along_the_line_simple(self, ccf, rrf, padding):
        row = self.analysed_map[ccf, rrf]
        row.shape = (row.shape[0], 1)
        if padding == 0:
            temp_row_full = row
        else:
            temp_row_full = binary_dilation(row, selem=np.ones((padding, padding)))

        temp_row_full = temp_row_full * 1
        temp_row_cut = temp_row_full.copy()
        l_slice_ids = []
        pt = 0
        for i, t in enumerate(temp_row_cut):
            if t == 0 and pt == 0:
                pt = t
            elif pt == 0 and t != 0:
                ts = []
                ts.append(i)
                pt = t
            elif pt != 0 and t != 0:
                ts.append(i)
                pt = t
            elif t == 0 and pt != 0:
                l_slice_ids.append(ts)
                ts = []
                pt = t
        return l_slice_ids, temp_row_full, temp_row_cut

    def __get_slices_along_the_line_kde(self, ccf, rrf, bandwidth, max_len, cutoff_percent):
        # advanced hypothesis generation
        row = self.analysed_map[ccf, rrf] * 1
        t_row = np.ones(row.shape) - row
        d_row = ndimage.distance_transform_cdt(t_row)
        d_row = max_len - d_row
        d_row = d_row.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(d_row)
        # cut the gaps
        temp_row_full = np.exp(kde.score_samples(d_row))
        temp_row_cut = temp_row_full.copy()
        temp_row_cut[temp_row_full < cutoff_percent * min(np.exp(kde.score_samples(d_row)))] = 0
        l_slice_ids = []
        pt = 0
        for i, t in enumerate(temp_row_cut):
            if t == 0 and pt == 0:
                pt = t
            elif pt == 0 and t != 0:
                ts = []
                ts.append(i)
                pt = t
            elif pt != 0 and t != 0:
                ts.append(i)
                pt = t
            elif t == 0 and pt != 0:
                l_slice_ids.append(ts)
                ts = []
                pt = t
        return l_slice_ids, temp_row_full, temp_row_cut, d_row

    def __estimate_wall_thickness_in_direction(self, lines_long, max_len, padding, V):
        thickness = []
        for l in lines_long:
            for s in np.arange(-1 * max_len, max_len, 1):
                rr, cc, rr_flag, cc_flag, flag = self.__get_cell_ids(V, l, s)
                if np.sum(self.analysed_map[cc[flag], rr[flag]] * 1) > 1:
                    l_slice_ids, _, _ = self.__get_slices_along_the_line_simple(cc[flag], rr[flag], padding)

                    for tslice in l_slice_ids:
                        thickness.append(len(tslice))
        return thickness

    def __generate_initial_hypothesis_direction_simple(self, lines_long, max_len, padding, cell_tr, vert):
        d_row_ret = []
        slices_ids = []
        slices = []
        cell_hypothesis = []
        lines_hypothesis = []
        kde_hypothesis = []
        kde_hypothesis_cut = []
        slices_dir = []

        for line_long in lines_long:
            temp_slice = []
            for s in np.arange(-1 * max_len, max_len, 1):
                rr, cc, rr_flag, cc_flag, flag = self.__get_cell_ids(vert, line_long, s)
                new_row = True
                if np.sum(self.analysed_map[cc[flag], rr[flag]] * 1) > 1:
                    l_slice_ids, temp_row_full, temp_row_cut = self.__get_slices_along_the_line_simple(cc[flag],
                                                                                                       rr[flag],
                                                                                                       padding)
                    l_slices_ids, l_slices, l_cell_hypothesis, l_lines_hypothesis, l_kde_hypothesis, l_kde_hypothesis_cut, l_temp_slice, new_row = self.__slice_wall(
                        cc, rr, flag, l_slice_ids, cell_tr, new_row, vert, line_long, temp_row_full, temp_row_cut, s)
                    slices_ids.extend(l_slices_ids)
                    slices.extend(l_slices)
                    cell_hypothesis.extend(l_cell_hypothesis)
                    lines_hypothesis.extend(l_lines_hypothesis)
                    kde_hypothesis.extend(l_kde_hypothesis)
                    kde_hypothesis_cut.extend(l_kde_hypothesis_cut)
                    temp_slice.extend(l_temp_slice)

            slices_dir.append(temp_slice)
        return d_row_ret, slices_ids, slices, cell_hypothesis, \
               lines_hypothesis, kde_hypothesis, kde_hypothesis_cut, slices_dir

    # Public Methods

    def process_map(self):
        self.__compute_fft()
        self.__get_dominant_directions()

        ti = time.time()
        if not self.comp:
            pass
        else:
            diag = 10
            mask_all = np.zeros(self.norm_ft_image.shape)
            min_l = self.map_size * (-1 / 2)
            max_l = self.map_size * (3 / 2)

            for p in self.comp:
                x1, y1 = he.pol2cart(diag, self.angles[p[0]] + np.pi / 2.0)
                x2, y2 = he.pol2cart(diag, self.angles[p[1]] + np.pi / 2.0)

                x1 = x1 + self.map_size / 2.0
                x2 = x2 + self.map_size / 2.0

                y1 = y1 + self.map_size / 2.0
                y2 = y2 + self.map_size / 2.0

                a = y2 - y1
                b = x1 - x2
                c = a * x1 + b * y1
                c1 = c + self.par
                c2 = c - self.par

                par = {'gamma': c, 'alpha': a, 'beta': b}
                #
                var = [min_l, max_l]
                X1_l, Y1_l, X2_l, Y2_l = he.lin_eq(par, var)
                #
                var = [0, self.map_size]
                X1, Y1, X2, Y2 = he.lin_eq(par, var)
                #
                par['gamma'] = c1
                X1_1, Y1_1, X2_1, Y2_1 = he.lin_eq(par, var)
                #
                par['gamma'] = c2
                X1_2, Y1_2, X2_2, Y2_2 = he.lin_eq(par, var)
                #
                Y_org = Y1
                if np.abs(Y_org) > 3 * self.map_size:
                    par = {'gamma': c, 'alpha': b, 'beta': a}
                    #
                    var = [min_l, max_l]
                    Y1_l, X1_l, Y2_l, X2_l = he.lin_eq(par, var)
                    #
                    var = [0, self.map_size]
                    Y1, X1, Y2, X2 = he.lin_eq(par, var)
                    #
                    par['gamma'] = c1
                    Y1_1, X1_1, Y2_1, X2_1 = he.lin_eq(par, var)
                    #
                    par['gamma'] = c2
                    Y1_2, X1_2, Y2_2, X2_2 = he.lin_eq(par, var)
                    #
                if max(X1_l, X2_l) < max(Y1_l, Y2_l):
                    self.lines_long_v.append([X1_l, Y1_l, X2_l, Y2_l])
                else:
                    self.lines_long_h.append([X1_l, Y1_l, X2_l, Y2_l])

                self.lines.append([X1, Y1, X2, Y2])
                mask_l = self.__generate_mask(X1_1, Y1_1, X2_1, Y2_1, X1_2, Y1_2, X2_2, Y2_2, Y_org)
                if not np.any(mask_l == 1):
                    mask_l = self.__generate_mask(X1_2, Y1_2, X2_2, Y2_2, X1_1, Y1_1, X2_1, Y2_1, Y_org)

                self.part_mask.append(mask_l)
                l_mask_ftimage = self.ft_image * mask_l
                l_mask_iftimage = np.fft.ifft2(l_mask_ftimage)
                self.part_reconstruction.append(np.abs(l_mask_iftimage))
                l_map_scored_good = np.abs(l_mask_iftimage) * (self.binary_map * 1)
                self.part_score.append(l_map_scored_good)

                mask_all = np.logical_or(mask_all, mask_l)

            mask_all = np.flipud(mask_all)
            mask_all_inv = np.ones(mask_all.shape)
            mask_all_inv[mask_all == 1] = 0
            logging.debug("Map scored in : %.2f s", time.time() - ti)

            self.mask_ft_image = self.ft_image * mask_all
            mask_iftimage = np.fft.ifft2(self.mask_ft_image)

            self.map_scored_good = np.abs(mask_iftimage) * (self.binary_map * 1)

    def simple_filter_map(self, tr):
        ti = time.time()
        l_map = np.array(np.abs(self.map_scored_good) / np.max(np.abs(self.map_scored_good)))
        self.quality_threshold = tr
        self.analysed_map = self.binary_map.copy()
        self.analysed_map[l_map < self.quality_threshold] = 0.0
        logging.debug("Map filtered simple in : %.2f s", time.time() - ti)

    def histogram_filtering(self):
        ti = time.time()
        pixels = np.abs(self.map_scored_good[self.binary_map > 0])

        self.quality_threshold, self.pixel_quality_gmm = self.__get_gmm_threshold(pixels)
        self.pixel_quality_histogram = self.__get_histogram(pixels)

        self.analysed_map = self.binary_map.copy()
        self.analysed_map[np.abs(self.map_scored_good) < self.quality_threshold] = 0.0
        logging.debug("Map filtered with histogram in : %.2f s", time.time() - ti)

    def generate_initial_hypothesis(self, **kwargs):
        if 'type' in kwargs:
            gen_type = kwargs["type"]
        else:
            gen_type = kwargs["type"]
        if 'min_wall' in kwargs:
            cell_tr = kwargs['min_wall']
        else:
            cell_tr = self.__estimate_wall_thickness()
        logging.info("Min wall thickness: %.2f", cell_tr)
        max_len = 5000
        t = time.time()
        if gen_type is 'kde':
            bandwidth = 0.00005
            cutoff_percent = 15
            self.d_row_v, self.slices_v_ids, self.slices_v, self.cell_hypothesis_v, self.lines_hypothesis_v, self.scored_hypothesis_v, self.scored_hypothesis_v_cut, self.slices_v_dir = self.__generate_initial_hypothesis_direction_with_kde(
                self.lines_long_v, max_len, bandwidth, cutoff_percent, cell_tr, True)
            self.d_row_h, self.slices_h_ids, self.slices_h, self.cell_hypothesis_h, self.lines_hypothesis_h, self.scored_hypothesis_h, self.scored_hypothesis_h_cut, self.slices_h_dir = self.__generate_initial_hypothesis_direction_with_kde(
                self.lines_long_h, max_len, bandwidth, cutoff_percent, cell_tr, False)
            logging.debug("Initial hypothesis generated with kde in %.2f s", time.time() - t)
        if gen_type is 'simple':
            padding = 1
            self.d_row_v, self.slices_v_ids, self.slices_v, self.cell_hypothesis_v, self.lines_hypothesis_v, self.scored_hypothesis_v, self.scored_hypothesis_v_cut, self.slices_v_dir = self.__generate_initial_hypothesis_direction_simple(
                self.lines_long_v, max_len, padding, cell_tr, True)
            self.d_row_h, self.slices_h_ids, self.slices_h, self.cell_hypothesis_h, self.lines_hypothesis_h, self.scored_hypothesis_h, self.scored_hypothesis_h_cut, self.slices_h_dir = self.__generate_initial_hypothesis_direction_simple(
                self.lines_long_h, max_len, padding, cell_tr, False)
            logging.debug("Initial hypothesis generated simple in %.2f", time.time() - t)

    def find_walls_flood_filing(self):
        overlap_ratio = 0.8
        ids = 2
        segments_in_directions = []
        for s in self.slices_v_dir:
            l_s = []
            temp_map = np.zeros(self.binary_map.shape)
            for p in s:
                for q in zip(p[0], p[1]):
                    temp_map[q[0], q[1]] = 1
            temp_map_fill = temp_map.copy()
            filled = False
            while not filled:
                seed = np.argwhere(temp_map_fill == 1)
                if seed.size != 0:
                    temp_map_fill = flood_fill(temp_map_fill, (seed[0][0], seed[0][1]), ids)
                    ids = ids + 1
                    cluster = np.where(temp_map_fill == ids - 1)
                    cluster = np.column_stack((cluster[0], cluster[1]))
                    d_cl = {"id": ids, "cells": list(map(tuple, cluster))}
                    l_s.append(d_cl)
                else:
                    filled = True
            segments_in_directions.append(l_s)
        for s in self.slices_h_dir:
            l_s = []
            temp_map = np.zeros(self.binary_map.shape)
            for p in s:
                for q in zip(p[0], p[1]):
                    temp_map[q[0], q[1]] = 1
            temp_map_fill = temp_map.copy()
            filled = False
            while not filled:
                seed = np.argwhere(temp_map_fill == 1)
                if seed.size != 0:
                    temp_map_fill = flood_fill(temp_map_fill, (seed[0][0], seed[0][1]), ids)
                    ids = ids + 1
                    cluster = np.where(temp_map_fill == ids - 1)
                    cluster = np.column_stack((cluster[0], cluster[1]))
                    d_cl = {"id": ids, "cells": list(map(tuple, cluster))}
                    l_s.append(d_cl)
                else:
                    filled = True
            segments_in_directions.append(l_s)
        segments = []
        self.overlap_score = []

        for dir in segments_in_directions:
            for c in dir:
                local_segment = WallSegment()
                local_segment.add_cells(c['cells'])
                local_segment.compute_central_lines()
                local_segment.id = c['id']
                segments.append(local_segment)

        done_segments = []
        overlap_list = []
        for c1 in segments:
            c1_disjoint = True
            for c2 in segments:
                if not c1 is c2:
                    if not c1.minimum_rotated_rectangle.disjoint(c2.minimum_rotated_rectangle):
                        c1_disjoint = False
                        overlap_list.append({c1.id, c2.id})
            if c1_disjoint:
                done_segments.append(c1)
        overlap_list = he.tuple_list_merger(overlap_list)

        import matplotlib.pyplot as plt

        logging.debug("overlap list:")
        logging.debug(overlap_list)
        logging.debug([len(x) for x in overlap_list])

        logging.debug("done segments:")
        logging.debug([x.id for x in done_segments])

        remove_list = []
        debug_it = 0
        for overlap in overlap_list:
            li = list(overlap)
            l = []
            for ll in li:
                l.append(next(item for item in segments if item.id == ll))
                print(l[-1].cells)
            inter, proj, intersections, intersection_ratios, merge = self.__process_wall_cluster(l, overlap_ratio)
            done, rm_list = self.__merge_walls(l, merge)

            done_segments.extend(done)
            remove_list.extend(rm_list)

            plt.figure()
            debug_it += 1
            print(debug_it)
            plt.imshow(self.binary_map, cmap="gray")
            for ll in l:
                y, x = ll.minimum_rotated_rectangle.exterior.xy
                plt.plot(x, y)

                y, x = ll.central_lines['short'].xy
                plt.plot(x, y)
            for ws1_id, ws1 in enumerate(l):
                for ws2_id, ws2 in enumerate(l):
                    if proj[ws1_id][ws2_id] is not None:
                        y, x = proj[ws1_id][ws2_id].xy
                        plt.plot(x, y, '--')
            for seg in rm_list:
                y, x = seg.minimum_rotated_rectangle.exterior.xy
                plt.plot(x, y, 'r')
            for seg in done:
                y, x = seg.minimum_rotated_rectangle.exterior.xy
                plt.plot(x, y, 'g')

        plt.figure()
        plt.imshow(self.binary_map, cmap="gray")
        for seg in segments:
            y, x = seg.minimum_rotated_rectangle.exterior.xy
            plt.plot(x, y, 'b')

        for seg in remove_list:
            y, x = seg.minimum_rotated_rectangle.exterior.xy
            plt.plot(x, y, 'r')

        for seg in done_segments:
            y, x = seg.minimum_rotated_rectangle.exterior.xy
            plt.plot(x, y, 'g')

        plt.show()

    def find_walls_flood_filing_with_overlaps(self):
        t = time.time()
        ids = 2
        for s in self.slices_v_dir:
            local_segments = []
            temp_map = np.zeros(self.binary_map.shape)
            for p in s:
                for q in zip(p[0], p[1]):
                    temp_map[q[0], q[1]] = 1
            temp_map_fill = temp_map.copy()
            filled = False
            while not filled:
                seed = np.argwhere(temp_map_fill == 1)
                if seed.size != 0:
                    temp_map_fill = flood_fill(temp_map_fill, (seed[0][0], seed[0][1]), ids)
                    ids = ids + 1
                else:
                    filled = True
                local_segment = sh.Segment()
                cluster = np.where(temp_map_fill == ids - 1)
                cluster = np.column_stack((cluster[0], cluster[1]))
                local_segment.add_cells(cluster)
                local_segment.compute_hull()
                local_segment.compute_mbb()
                local_segment.id = ids
                local_segments.append(local_segment)
            self.segments_v.append(local_segments)

            local_mbb_lines = []
            for l_segment in local_segments:
                x1, y1 = l_segment.center
                x2, y2 = l_segment.center[0] + l_segment.rectangle_direction[0], l_segment.center[1] + \
                         l_segment.rectangle_direction[1]
                a = y2 - y1
                b = x1 - x2
                c = a * (x1) + b * (y1)
                if not b == 0:
                    X1 = 0
                    Y1 = (c - a * X1) / b
                    X2 = self.binary_map.shape[0]
                    Y2 = (c - a * X2) / b
                if np.abs(Y1) > 3 * np.max(self.binary_map.shape) or b == 0:
                    #
                    Y1 = 0
                    X1 = (c - b * Y1) / a
                    Y2 = self.binary_map.shape[1]
                    X2 = (c - b * Y2) / a
                    #
                local_mbb_lines.append({"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2})
                self.all_lines.append(((X1, Y1), (X2, Y2)))
            self.segments_v_mbb_lines.append(local_mbb_lines)

        for s in self.slices_h_dir:
            local_segments = []
            temp_map = np.zeros(self.binary_map.shape)
            for p in s:
                for q in zip(p[0], p[1]):
                    temp_map[q[0], q[1]] = 1
            temp_map_fill = temp_map.copy()
            filled = False
            while not filled:
                seed = np.argwhere(temp_map_fill == 1)
                if seed.size != 0:
                    temp_map_fill = flood_fill(temp_map_fill, (seed[0][0], seed[0][1]), ids)
                    ids = ids + 1
                else:
                    filled = True
                local_segment = sh.Segment()
                cluster = np.where(temp_map_fill == ids - 1)
                cluster = np.column_stack((cluster[0], cluster[1]))
                local_segment.add_cells(cluster)
                local_segment.compute_hull()
                local_segment.compute_mbb()
                local_segment.id = ids
                local_segments.append(local_segment)
            self.segments_h.append(local_segments)
            local_mbb_lines = []
            for l_segment in local_segments:
                x1, y1 = l_segment.center
                x2, y2 = l_segment.center[0] + l_segment.rectangle_direction[0], l_segment.center[1] + \
                         l_segment.rectangle_direction[1]
                a = y2 - y1
                b = x1 - x2
                c = a * (x1) + b * (y1)
                if not b == 0:
                    X1 = 0
                    Y1 = (c - a * X1) / b
                    X2 = self.binary_map.shape[0]
                    Y2 = (c - a * X2) / b
                if np.abs(Y1) > 3 * np.max(self.binary_map.shape) or b == 0:
                    #
                    Y1 = 0
                    X1 = (c - b * Y1) / a
                    Y2 = self.binary_map.shape[1]
                    X2 = (c - b * Y2) / a
                    #
                local_mbb_lines.append({"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2})
                self.all_lines.append(((X1, Y1), (X2, Y2)))
            self.segments_h_mbb_lines.append(local_mbb_lines)
        self.all_lines = list(dict.fromkeys(self.all_lines))

        logging.debug("Found walls with flood filling in: %.2f", time.time() - t)

    def find_walls_with_line_segments(self):
        t = time.time()
        eps = 10
        min_samples = 2
        if len(self.slices_h_dir) is not 0:
            for direction in self.slices_h_dir:
                slice_lines = self.__generate_line_segments_per_direction(direction)
                clustering_h = DBSCAN(eps=eps, min_samples=min_samples,
                                      metric=he.shortest_distance_between_segements).fit(
                    slice_lines)
                self.slice_h_lines.append(slice_lines)
                self.clustering_h_labels.append(clustering_h.labels_)

        if len(self.slices_v_dir) is not 0:
            for direction in self.slices_v_dir:
                slice_lines = self.__generate_line_segments_per_direction(direction)
                clustering_v = DBSCAN(eps=eps, min_samples=min_samples,
                                      metric=he.shortest_distance_between_segements).fit(
                    slice_lines)
                self.slice_v_lines.append(slice_lines)
                self.clustering_v_labels.append(clustering_v.labels_)

        id = 2
        temp_map = np.zeros(self.binary_map.shape)

        last_label = 0
        for slice, label in zip(self.slices_h_dir, self.clustering_h_labels):
            if last_label != label:
                id = id + 1
            for s in zip(slice[0][0], slice[1][0]):
                temp_map[s[0]][s[1]] = id
            last_label = label
        last_label = 0
        for slice, label in zip(self.slices_v_dir, self.clustering_v_labels):
            if last_label != label:
                id = id + 1
            for s in zip(slice[0][0], slice[1][0]):
                temp_map[s[0]][s[1]] = id
            last_label = label
        self.labeled_map_line_segment = temp_map
        logging.debug("Found walls with line clustering in: %.2f", time.time() - t)
