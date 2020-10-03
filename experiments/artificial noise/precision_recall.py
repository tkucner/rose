import ast
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

# 1) list all the files in the directory
result_dir = "results"

result_files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]

histogram_result_file = "rose_scoring_histogram_threshold.tex"
with open(join(result_dir, histogram_result_file)) as f:
    lines = f.read().splitlines()

data_lines_histogram = [ast.literal_eval((("{\"name:\"" + l.strip().split(" ", maxsplit=1)[0] + "\" " +
                                           l.strip().split(" ", maxsplit=1)[1] + "}").replace(" ", ", \"")).replace(":",
                                                                                                                    "\" : "))
                        for l in lines]

# 2) parse the files
for rf in result_files:
    # find histogram data
    dl = [d if d['name'].split(".", maxsplit=1)[0] == rf.split(".", maxsplit=1)[0] else None for d in
          data_lines_histogram]
    dl = list(filter(None, dl))

    with open(join(result_dir, rf)) as f:
        lines = f.read().splitlines()
    name = rf.strip().split(".", maxsplit=1)[0]
    data = [ast.literal_eval(
        "{\"" + ((l.strip().split(" ", maxsplit=1)[1]).replace(" ", ", \"")).replace(":", "\" : ") + "}") for l in
        lines]
    headers = ['threshold', 'precision', 'recall', 'true_negative_rate', 'true_positive', 'true_negative',
               'false_positive', 'false_negative']
    plot_data = {}
    for h in headers:
        plot_data.update({h: [d[h] for d in data]})

    fig = plt.figure(constrained_layout=True, figsize=[16, 9], dpi=300)

    gs = fig.add_gridspec(2, 12)
    f3_ax1 = fig.add_subplot(gs[0, 0:3])
    f3_ax1.set_title('Precision')
    f3_ax1.plot(plot_data['threshold'], plot_data['precision'], '.')
    if len(dl) == 1:
        f3_ax1.plot(dl[0]['threshold'], dl[0]['precision'], '*')
    f3_ax1.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

    f3_ax2 = fig.add_subplot(gs[0, 3:6])
    f3_ax2.set_title('Recall')
    f3_ax2.plot(plot_data['threshold'], plot_data['recall'], '.')
    if len(dl) == 1:
        f3_ax2.plot(dl[0]['threshold'], dl[0]['recall'], '*')
    f3_ax2.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

    f3_ax3 = fig.add_subplot(gs[0, 6:9])
    f3_ax3.set_title('Selectivity')
    f3_ax3.plot(plot_data['threshold'], plot_data['true_negative_rate'], '.')
    if len(dl) == 1:
        f3_ax3.plot(dl[0]['threshold'], dl[0]['true_negative_rate'], '*')
    f3_ax3.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

    f3_ax8 = fig.add_subplot(gs[0, 9:12])
    f3_ax8.set_title('ROC')
    f3_ax8.plot(1 - np.array(plot_data['true_negative_rate']), plot_data['recall'], '.')
    if len(dl) == 1:
        f3_ax8.plot(1 - dl[0]['true_negative_rate'], dl[0]['recall'], '*')
    f3_ax8.plot([-0.1, 1.1], [-0.1, 1.1], '--')
    f3_ax8.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

    f3_ax4 = fig.add_subplot(gs[1, 0:3])
    f3_ax4.set_title('True Positive')
    f3_ax4.plot(plot_data['threshold'], plot_data['true_positive'], '.')

    f3_ax5 = fig.add_subplot(gs[1, 3:6])
    f3_ax5.set_title('False Positive')
    f3_ax5.plot(plot_data['threshold'], plot_data['false_positive'], '.')

    f3_ax6 = fig.add_subplot(gs[1, 6:9])
    f3_ax6.set_title('True Negative')
    f3_ax6.plot(plot_data['threshold'], plot_data['true_negative'], '.')

    f3_ax7 = fig.add_subplot(gs[1, 9:12])
    f3_ax7.set_title('False Negative')
    f3_ax7.plot(plot_data['threshold'], plot_data['false_negative'], '.')
    fig.suptitle(name)
    fig.savefig(join("plots", name + ".png"))
