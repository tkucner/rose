import pickle
import statistics as stat

import numpy as np


def min_wraped_distance(x1, x2):
    dx1 = abs(x1 - x2)
    dx2 = np.pi
    if (dx1 > np.pi / 2):
        dx2 = np.pi - dx1
    return min(dx1, dx2)


directory = "results"
fiels_to_analise = ['dir_batches_Hough_90.pkl', 'dir_error_types_Hough_90.pkl']

file = open(directory + "/" + fiels_to_analise[0], 'rb')
maps = pickle.load(file)
file = open(directory + "/" + fiels_to_analise[1], 'rb')
stats = pickle.load(file)

ref_angles = [[0, np.pi / 2] for x in range(100)]

ec = 0

ec = 0
errors = []
# error_names = ["obs_count", "line_err", "ang_error", "laser_noise"]
# error_types = {i: set() for i in error_names}

for ra, m in zip(ref_angles, maps):
    for variant in m['variations']:
        mr = []
        mrf = []
        for d in variant['directions']:
            min_err = np.pi
            for r in ra:
                min_err = min(min_err, min_wraped_distance(d, r))
            mr.append(min_err)

        dr = (abs(len(ra) - len(variant['directions'])))

        lv = {'id': variant['id'], 'obs_count': variant['obs_count'], 'line_err': variant['line_err'],
              'ang_error': variant['ang_error'], 'laser_noise': variant['laser_noise'], 'dr': dr, 'ang_err': mr}
        errors.append(lv)

# zero the stats array
for ks, s in stats.items():
    for ket, et in s.items():
        for kv, v in et.items():
            stats[ks][ket][kv] = []

for err in errors:
    if err['obs_count'] != 0.0:
        stats['obs_count'][err['obs_count']]['ang_errors'].extend(err['ang_err'])
        stats['obs_count'][err['obs_count']]['directions_count_error'].append(err['dr'])
    if err['line_err'] != 0.0:
        stats['line_err'][err['line_err']]['ang_errors'].extend(err['ang_err'])
        stats['line_err'][err['line_err']]['directions_count_error'].append(err['dr'])
    if err['ang_error'] != 0.0:
        stats['ang_error'][err['ang_error']]['ang_errors'].extend(err['ang_err'])
        stats['ang_error'][err['ang_error']]['directions_count_error'].append(err['dr'])
    if err['laser_noise'] != 0.0:
        stats['laser_noise'][err['laser_noise']]['ang_errors'].extend(err['ang_err'])
        stats['laser_noise'][err['laser_noise']]['directions_count_error'].append(err['dr'])
for ks, s in stats.items():
    for ket, et in s.items():
        et["stats_ang"] = {"mean": stat.mean(et["ang_errors"]), "std": stat.pstdev(et["ang_errors"])}
        et["stats_directions_count"] = {"mean": stat.mean(et["directions_count_error"]),
                                        "std": stat.pstdev(et["directions_count_error"])}

for ket, et in stats.items():
    print(ket)
    for kv, v in et.items():
        print("{:.2f}:  angular M={:.3f} std={:.3f}, directions M={:.3f} std={:.3f}".format(kv, v["stats_ang"]["mean"],
                                                                                            v["stats_ang"]["std"],
                                                                                            v["stats_directions_count"][
                                                                                                "mean"],
                                                                                            v["stats_directions_count"][
                                                                                                "std"]))

#
#
#
# for ra, m in zip(ref_angles,maps):
#     print("----------------------------")
#     for variant in m['variations']:
#         mr=[]
#         mrf=[]
#         for d in variant['directions']:
#             min_err = np.pi
#             for r in ra:
#                 min_err = min(min_err, min_wraped_distance(d, r))
#
#             mr.append(min_err)
#             mrf.append(min_err>0.01)
#         dr=(abs(len(ra) - len(variant['directions'])))
#         if dr>0 and any(mrf):
#             print(variant['id'])
#             print(variant['obs_count'],variant['line_err'],variant['ang_error'],variant['laser_noise'])
#             print(dr)
#             print(mr)
#             ec+=1
# print("total error count: {}".format(ec))
#
# for ket, et in stats.items():
#     print(ket)
#     for kv, v in et.items():
#         print("{:.2f}:  angular M={:.3f} std={:.3f}, directions M={:.3f} std={:.3f}".format(kv, v["stats_ang"]["mean"],
#                                                                                             v["stats_ang"]["std"],
#                                                                                             v["stats_directions_count"][
#                                                                                                 "mean"],
#                                                                                             v["stats_directions_count"][
#                                                                                                 "std"]))


# print("directions_count_error")
# for key_error, item_error in stats.items():
#     print(key_error)
#     for key_value, item_value in item_error.items():
#         print("    {}".format(key_value))
#         for it, val in enumerate(item_value["directions_count_error"]):
#             if val > 0:
#                 print("           {}->{}".format(it, val))
#                 print(maps[it])
