import argparse

from skimage import io
from skimage.util import img_as_ubyte

from fft_structure_extraction import FFTStructureExtraction as structure_extraction

if __name__ == "__main__":
    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file', help='Path to the map')
    args = parser.parse_args()

    # FFT

    grid_map = img_as_ubyte(io.imread(args.img_file))
    rose = structure_extraction(grid_map, peak_height=0.2, par=50)
    rose.process_map()

    filter_level = 0.18
    # rose.simple_filter_map(filter_level)

    # rose.histogram_filtering()

    #rose.generate_initial_hypothesis_with_kde()
    rose.generate_initial_hypothesis_simple()
    # rose.find_walls_with_line_segments()
    rose.find_walls_flood_filing()

    rose.report()

    visualisation = {"Binary map": True,
                     "FFT Spectrum": False,
                     "Unfolded FFT Spectrum": True,
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
                     "Map with directions": False,
                     "Partial Scores": False,
                     "Partial Reconstructs": False,
                     "Threshold Setup with Clusters": False,
                     "Cluster Filtered Map": False,
                     "Map with walls": False,
                     "Map with slices": False,
                     "Wall lines from mbb": False,
                     "Labels and Raw map": False,
                     "Raw line segments": False,
                     "Clustered line segments": False,
                     "Short wall lines from mbb": True,
                     "Short wall lines over original map": True

                     }
    rose.show(visualisation)
