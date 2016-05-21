""" grab all the xval info and put it together """

import glob
import numpy as np

test_set_f = glob.glob("test_set*.npz")
test_results_f = glob.glob("test_results*.npz")

ids_all = []
labels_all = []

for f in test_set_f:
    ids = np.load(f)['arr_0']
    for val in ids:
        ids_all.append(val)

ids_all = np.array(ids_all)

for f in test_results_f:
    labels = np.load(f)['arr_0']
    for val in labels:
        labels_all.append(val)

labels_all = np.array(labels_all)

np.savez("test_ids_all.npz", ids_all)
np.savez("test_labels_all.npz", labels_all)
