import argparse

from skimage import io
from skimage.util import img_as_ubyte

from map_quality_fft import map_quality_fft as mq_fft

if __name__ == "__main__":
    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file', help='Path to the map')
    args = parser.parse_args()

    # FFT

    grid_map = img_as_ubyte(io.imread(args.img_file))
    mq = mq_fft(grid_map, peak_hight=0.2, par=50)
    mq.process_map()

    filter_level = 0.18
    mq.simple_filter_map(filter_level)

    mq.histogram_filtering()

    mq.generate_intiail_hypothesis_filtered()
    # mq.hypothesis_clustering()
    mq.find_walls_floodfiling()
    mq.find_walls_knn()

    mq.report()

    #     count, ps, isects, G, Q=pc.polygon_count(mq.all_lines)
    #
    #     flags = {
    #     "input": True,
    #     "graph": False,
    #     "result": False,
    #     "verbose": True
    # }
    #
    #     pc.show(mq.all_lines, ps, isects, G, Q, flags,count)
    ####################################################################################################################
    # VISUALISATION FFT
    ####################################################################################################################

    visualisation = {"Binary map": False,
                     "FFT Spectrum": False,
                     "Unfolded FFT Spectrum": False,
                     "FFT Spectrum Signal": False,
                     "FFT Spectrum Noise": False,
                     "Map Reconstructed Signal": False,
                     "Map Reconstructed Noise": False,
                     "Map Scored Good": False,
                     "Map Scored Bad": False,
                     "Map Scored Diff": False,
                     "Map Split Good": False,
                     "FFT Map Split Good": False,
                     "Side by Side": False,
                     "Histogram of pixels quality": False,
                     "Histogram of scaled pixels quality": False,
                     "Simple Filtered Map": False,
                     "FFT spectrum with directions": False,
                     "Map with directions": True,
                     "Partial Scores": False,
                     "Partial Reconstructs": False,
                     "Treshold Setup with Clusters": False,
                     "Cluster Filtered Map": False,
                     "Map with walls": False,
                     "Map with slices": True,
                     "Wall lines from mbb": True,
                     "Labels and Raw map": True
                     }
    mq.show(visualisation)
