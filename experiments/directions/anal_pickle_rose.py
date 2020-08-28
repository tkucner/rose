import pickle

directory = "results"
fiels_to_analise = ['dir_batches_FFT.pkl', 'dir_error_types_FFT.pkl']

file = open(directory + "/" + fiels_to_analise[0], 'rb')
maps = pickle.load(file)
file = open(directory + "/" + fiels_to_analise[1], 'rb')
stats = pickle.load(file)

print("directions_count_error")
for key_error, item_error in stats.items():
    print(key_error)
    for key_value, item_value in item_error.items():
        print("    {}".format(key_value))
        for it, val in enumerate(item_value["directions_count_error"]):
            if val > 0:
                print("           {}->{}".format(it, val))
                print(maps[it])
