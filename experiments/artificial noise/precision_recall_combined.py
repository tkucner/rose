import ast
from os import listdir
from os.path import isfile, join

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# 1) list all the files in the directory
result_dir = "results"
result_files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]


def get_params(c):
    c = c.strip().split(".", maxsplit=1)[0]
    c_split = c.rsplit("_", maxsplit=6)
    r = {"env": c_split[0], "count": int(c_split[2]), "size": int(c_split[4]), "type": c_split[6]}
    return r


histogram_result_file = "rose_scoring_histogram_threshold.txt"
with open(join(result_dir, histogram_result_file)) as f:
    lines = f.read().splitlines()

data_lines_histogram = [ast.literal_eval((("{\"name:\"" + l.strip().split(" ", maxsplit=1)[0] + "\" " +
                                           l.strip().split(" ", maxsplit=1)[1] + "}").replace(" ", ", \"")).replace(":",
                                                                                                                    "\" : "))
                        for l in lines]
for dt in data_lines_histogram:
    dt["params"] = get_params(dt["name"])
# 2) get unique params
unique = {"env": list(set([dlh["params"]["env"] for dlh in data_lines_histogram])),
          "type": list(set([dlh["params"]["type"] for dlh in data_lines_histogram])),
          "count": sorted(list(set([int(dlh["params"]["count"]) for dlh in data_lines_histogram]))),
          "size": sorted(list(set([int(dlh["params"]["size"]) for dlh in data_lines_histogram])))}

# 3) set order
order = ["env", "type", "count", "size"]

for o0 in unique[order[0]]:
    for o1 in unique[order[1]]:
        for o2 in unique[order[2]]:
            plotting_lines = [dlh for dlh in data_lines_histogram if (
                    (dlh["params"][order[0]] == o0) and (dlh["params"][order[1]] == o1) and (
                    dlh["params"][order[2]] == o2))]
            plotting_lines = sorted(plotting_lines, key=lambda i: i['params'][order[3]])
            # find result files
            sub_result_files = [rf["name"].strip().split(".", maxsplit=1)[0] + ".txt" for rf in plotting_lines]

            fig = plt.figure(constrained_layout=True, figsize=[16, 9], dpi=100)

            n = len(unique[order[3]])
            colors = cm.viridis(np.linspace(0, 1, n))

            save_name = order[0] + "_" + str(o0) + "_" + order[1] + "_" + str(o1) + "_" + order[2] + "_" + str(o2)
            gs = fig.add_gridspec(2, 14)
            f3_ax1 = fig.add_subplot(gs[0, 0:3])
            f3_ax1.set_title('Precision')
            f3_ax1.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

            f3_ax2 = fig.add_subplot(gs[0, 3:6])
            f3_ax2.set_title('Recall')
            f3_ax2.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

            f3_ax3 = fig.add_subplot(gs[0, 6:9])
            f3_ax3.set_title('Selectivity')
            f3_ax3.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

            f3_ax8 = fig.add_subplot(gs[0, 9:12])
            f3_ax8.set_title('ROC')
            f3_ax8.plot([-0.1, 1.1], [-0.1, 1.1], '--')
            f3_ax8.axis(ymin=-0.1, ymax=1.1, xmin=-0.1, xmax=1.1)

            f3_ax4 = fig.add_subplot(gs[1, 0:3])
            f3_ax4.set_title('True Positive')
            f3_ax5 = fig.add_subplot(gs[1, 3:6])
            f3_ax5.set_title('False Positive')
            f3_ax6 = fig.add_subplot(gs[1, 6:9])
            f3_ax6.set_title('True Negative')
            f3_ax7 = fig.add_subplot(gs[1, 9:12])
            f3_ax7.set_title('False Negative')

            i = 0
            for srf, dl in zip(sub_result_files, plotting_lines):
                if not isfile(join(result_dir, srf)):
                    continue
                with open(join(result_dir, srf)) as f:
                    lines = f.read().splitlines()
                name = srf.strip().split(".", maxsplit=1)[0]
                data = [ast.literal_eval(
                    "{\"" + ((l.strip().split(" ", maxsplit=1)[1]).replace(" ", ", \"")).replace(":", "\" : ") + "}")
                    for l
                    in
                    lines]
                headers = ['threshold', 'precision', 'recall', 'true_negative_rate', 'true_positive', 'true_negative',
                           'false_positive', 'false_negative']
                plot_data = {}
                for h in headers:
                    plot_data.update({h: [d[h] for d in data]})

                f3_ax1.plot(plot_data['threshold'], plot_data['precision'], '.-', color=colors[i],
                            label=dl["params"][order[3]])
                f3_ax1.plot(dl['threshold'], dl['precision'], '*r')

                f3_ax2.plot(plot_data['threshold'], plot_data['recall'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))
                f3_ax2.plot(dl['threshold'], dl['recall'], '*r')

                f3_ax3.plot(plot_data['threshold'], plot_data['true_negative_rate'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))
                f3_ax3.plot(dl['threshold'], dl['true_negative_rate'], '*r')

                f3_ax8.plot(1 - np.array(plot_data['true_negative_rate']), plot_data['recall'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))
                f3_ax8.plot(1 - dl['true_negative_rate'], dl['recall'], '*r')

                f3_ax4.plot(plot_data['threshold'], plot_data['true_positive'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))

                f3_ax5.plot(plot_data['threshold'], plot_data['false_positive'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))

                f3_ax6.plot(plot_data['threshold'], plot_data['true_negative'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))

                f3_ax7.plot(plot_data['threshold'], plot_data['false_negative'], '.-', color=colors[i],
                            label=str(dl["params"][order[3]]))

                i += 1
            lines, labels = f3_ax1.get_legend_handles_labels()

            fig.legend(lines, labels,
                       loc='right', title=order[3])

            print(save_name)
            fig.suptitle(save_name)
            fig.savefig(join("plots_combined", save_name + ".png"))
            # plt.show()
            plt.close()
