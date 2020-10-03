import statistics as stats

import matplotlib.pyplot as plt


def quantiles(data, *, n=4, method='exclusive'):
    """Divide *data* into *n* continuous intervals with equal probability.
    Returns a list of (n - 1) cut points separating the intervals.
    Set *n* to 4 for quartiles (the default).  Set *n* to 10 for deciles.
    Set *n* to 100 for percentiles which gives the 99 cuts points that
    separate *data* in to 100 equal sized groups.
    The *data* can be any iterable containing sample.
    The cut points are linearly interpolated between data points.
    If *method* is set to *inclusive*, *data* is treated as population
    data.  The minimum value is treated as the 0th percentile and the
    maximum value is treated as the 100th percentile.
    """
    if n < 1:
        raise stats.StatisticsError('n must be at least 1')
    data = sorted(data)
    ld = len(data)
    if ld < 2:
        raise stats.StatisticsError('must have at least two data points')
    if method == 'inclusive':
        m = ld - 1
        result = []
        for i in range(1, n):
            j = i * m // n
            delta = i * m - j * n
            interpolated = (data[j] * (n - delta) + data[j + 1] * delta) / n
            result.append(interpolated)
        return result
    if method == 'exclusive':
        m = ld + 1
        result = []
        for i in range(1, n):
            j = i * m // n  # rescale i to m/n
            j = 1 if j < 1 else ld - 1 if j > ld - 1 else j  # clamp to 1 .. ld-1
            delta = i * m - j * n  # exact integer math
            interpolated = (data[j - 1] * (n - delta) + data[j] * delta) / n
            result.append(interpolated)
        return result
    raise ValueError(f'Unknown method: {method!r}')


file = "logs/rose_2020_09_10_19_50_34.log"
key_word = "rose_scoring"
my_file = open(file, "r")
content = my_file.read()
content_list = content.split("\n")
content_list = [(":".join(c.split(":")[3:])).replace(".png", "") for c in content_list if key_word in c]
content_list = content_list[3:]
my_file.close()
content_list = [c.split(" ") for c in content_list]
list_ana = []
for c in content_list:
    ic = c[0].split("_")
    r = {ic[-2]: ic[-1], ic[-4]: int(ic[-3]), ic[-6]: int(ic[-5]), "env": "_".join(ic[:-6]),
         c[1].split(":")[0]: float(c[1].split(":")[1]), c[2].split(":")[0]: float(c[2].split(":")[1]),
         c[3].split(":")[0]: float(c[3].split(":")[1]), c[4].split(":")[0]: float(c[4].split(":")[1])}
    list_ana.append(r)
# compute statistics
# 1-> total stats

precisions = [c["precision"] for c in list_ana]
recalls = [c["recall"] for c in list_ana]
true_negative_rates = [c["true_negative_rate"] for c in list_ana]
total_stats = {
    "precision": {"av": stats.mean(precisions), "std": stats.stdev(precisions), "median": stats.median(precisions),
                  "min": min(precisions), "max": max(precisions), "q1": quantiles(precisions)[0],
                  "q3": quantiles(precisions)[2]},
    "recall": {"av": stats.mean(recalls), "std": stats.stdev(recalls), "median": stats.median(recalls),
               "min": min(recalls), "max": max(recalls), "q1": quantiles(recalls)[0],
               "q3": quantiles(recalls)[2]},
    "true negative rate": {"av": stats.mean(true_negative_rates), "std": stats.stdev(true_negative_rates),
                           "median": stats.median(true_negative_rates), "min": min(true_negative_rates),
                           "max": max(true_negative_rates), "q1": quantiles(true_negative_rates)[0],
                           "q3": quantiles(true_negative_rates)[2]},
}

# 2-> stats per environment
env_list = list(set([c["env"] for c in list_ana]))
env_stats = {}
data_recalls = []
data_precisions = []
for indicator in env_list:
    precisions = [c["precision"] for c in list_ana if c["env"] == indicator]
    recalls = [c["recall"] for c in list_ana if c["env"] == indicator]
    true_negative_rates = [c["true_negative_rate"] for c in list_ana if c["env"] == indicator]
    env_stats[indicator] = {
        "precision": {"av": stats.mean(precisions), "std": stats.stdev(precisions), "median": stats.median(precisions),
                      "min": min(precisions), "max": max(precisions), "q1": quantiles(precisions)[0],
                      "q3": quantiles(precisions)[2]},
        "recall": {"av": stats.mean(recalls), "std": stats.stdev(recalls), "median": stats.median(recalls),
                   "min": min(recalls), "max": max(recalls), "q1": quantiles(recalls)[0],
                   "q3": quantiles(recalls)[2]},
        "true negative rate": {"av": stats.mean(true_negative_rates), "std": stats.stdev(true_negative_rates),
                               "median": stats.median(true_negative_rates), "min": min(true_negative_rates),
                               "max": max(true_negative_rates), "q1": quantiles(true_negative_rates)[0],
                               "q3": quantiles(true_negative_rates)[2]},
    }
    data_recalls.append(recalls)
    data_precisions.append(precisions)
fig1, ax = plt.subplots(1, 2)
ax[0].boxplot(data_precisions)
ax[1].boxplot(data_recalls)
ax[0].set_title("precision")
ax[1].set_title("recall")
ax[0].set_title("precision")
ax[0].set_ylim([0, 1.2])
ax[1].set_ylim([0, 1.2])
ax[0].set_xticklabels(env_list)
ax[1].set_xticklabels(env_list)
fig1.suptitle("Environment")
plt.show()

# 3-> stats per obstacle count

ocount_list = list(set([c["ocount"] for c in list_ana]))
ocount_stats = {}
data_recalls = []
data_precisions = []
for indicator in ocount_list:
    precisions = [c["precision"] for c in list_ana if c["ocount"] == indicator]
    recalls = [c["recall"] for c in list_ana if c["ocount"] == indicator]
    true_negative_rates = [c["true_negative_rate"] for c in list_ana if c["ocount"] == indicator]
    ocount_stats[indicator] = {
        "precision": {"av": stats.mean(precisions), "std": stats.stdev(precisions), "median": stats.median(precisions),
                      "min": min(precisions), "max": max(precisions), "q1": quantiles(precisions)[0],
                      "q3": quantiles(precisions)[2]},
        "recall": {"av": stats.mean(recalls), "std": stats.stdev(recalls), "median": stats.median(recalls),
                   "min": min(recalls), "max": max(recalls), "q1": quantiles(recalls)[0],
                   "q3": quantiles(recalls)[2]},
        "true negative rate": {"av": stats.mean(true_negative_rates), "std": stats.stdev(true_negative_rates),
                               "median": stats.median(true_negative_rates), "min": min(true_negative_rates),
                               "max": max(true_negative_rates), "q1": quantiles(true_negative_rates)[0],
                               "q3": quantiles(true_negative_rates)[2]},
    }
    data_recalls.append(recalls)
    data_precisions.append(precisions)
fig1, ax = plt.subplots(1, 2)
ax[0].boxplot(data_precisions)
ax[1].boxplot(data_recalls)
ax[0].set_title("precision")
ax[1].set_title("recall")
ax[0].set_ylim([0, 1.2])
ax[1].set_ylim([0, 1.2])
ax[0].set_xticklabels(ocount_list)
ax[1].set_xticklabels(ocount_list)
fig1.suptitle("Obstacle count")
plt.show()

# 4-> stats per obstacle count

osize_list = list(set([c["osize"] for c in list_ana]))
osize_stats = {}
data_recalls = []
data_precisions = []
for indicator in osize_list:
    precisions = [c["precision"] for c in list_ana if c["osize"] == indicator]
    recalls = [c["recall"] for c in list_ana if c["osize"] == indicator]
    true_negative_rates = [c["true_negative_rate"] for c in list_ana if c["osize"] == indicator]
    osize_stats[indicator] = {
        "precision": {"av": stats.mean(precisions), "std": stats.stdev(precisions), "median": stats.median(precisions),
                      "min": min(precisions), "max": max(precisions), "q1": quantiles(precisions)[0],
                      "q3": quantiles(precisions)[2]},
        "recall": {"av": stats.mean(recalls), "std": stats.stdev(recalls), "median": stats.median(recalls),
                   "min": min(recalls), "max": max(recalls), "q1": quantiles(recalls)[0],
                   "q3": quantiles(recalls)[2]},
        "true negative rate": {"av": stats.mean(true_negative_rates), "std": stats.stdev(true_negative_rates),
                               "median": stats.median(true_negative_rates), "min": min(true_negative_rates),
                               "max": max(true_negative_rates), "q1": quantiles(true_negative_rates)[0],
                               "q3": quantiles(true_negative_rates)[2]},
    }
    data_recalls.append(recalls)
    data_precisions.append(precisions)
fig1, ax = plt.subplots(1, 2)
ax[0].boxplot(data_precisions)
ax[1].boxplot(data_recalls)
ax[0].set_title("precision")
ax[1].set_title("recall")
ax[0].set_ylim([0, 1.2])
ax[1].set_ylim([0, 1.2])
ax[0].set_xticklabels(osize_list)
ax[1].set_xticklabels(osize_list)
fig1.suptitle("Obstacle size")
plt.show()
# 4-> stats per obstacle type

otype_list = list(set([c["otype"] for c in list_ana]))
otype_stats = {}
data_recalls = []
data_precisions = []
for indicator in otype_list:
    precisions = [c["precision"] for c in list_ana if c["otype"] == indicator]
    recalls = [c["recall"] for c in list_ana if c["otype"] == indicator]
    true_negative_rates = [c["true_negative_rate"] for c in list_ana if c["otype"] == indicator]
    otype_stats[indicator] = {
        "precision": {"av": stats.mean(precisions), "std": stats.stdev(precisions), "median": stats.median(precisions),
                      "min": min(precisions), "max": max(precisions), "q1": quantiles(precisions)[0],
                      "q3": quantiles(precisions)[2]},
        "recall": {"av": stats.mean(recalls), "std": stats.stdev(recalls), "median": stats.median(recalls),
                   "min": min(recalls), "max": max(recalls), "q1": quantiles(recalls)[0],
                   "q3": quantiles(recalls)[2]},
        "true negative rate": {"av": stats.mean(true_negative_rates), "std": stats.stdev(true_negative_rates),
                               "median": stats.median(true_negative_rates), "min": min(true_negative_rates),
                               "max": max(true_negative_rates), "q1": quantiles(true_negative_rates)[0],
                               "q3": quantiles(true_negative_rates)[2]},
    }
    data_recalls.append(recalls)
    data_precisions.append(precisions)
fig1, ax = plt.subplots(1, 2)
ax[0].boxplot(data_precisions)
ax[1].boxplot(data_recalls)
ax[0].set_title("precision")
ax[1].set_title("recall")
ax[0].set_ylim([0, 1.2])
ax[1].set_ylim([0, 1.2])
ax[0].set_xticklabels(otype_list)
ax[1].set_xticklabels(otype_list)
fig1.suptitle("Obstacle Type")
plt.show()
