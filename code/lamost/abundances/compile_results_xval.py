""" Compile the results from the cross-validation """

import numpy as np

ngroups = 8
#direc_old = "/Users/annaho/Data/LAMOST/Abundances/with_col_mask/xval_with_cuts_2"
# for astro-node2
direc_old = "/home/annaho/TheCannon/data/LAMOST/abundances"
direc = "."

ref_id = np.load("%s/ref_id.npz" %direc_old)['arr_0']
groups = np.load("%s/assignments.npz" %direc_old)['arr_0']
ref_label = np.load("%s/ref_label.npz" %direc_old)['arr_0']
num_obj, num_label = ref_label.shape

tr_label_vals = np.zeros((num_obj, num_label))
all_cannon_label_vals = np.zeros((num_obj, num_label))
all_cannon_label_errs = np.zeros(all_cannon_label_vals.shape)
all_cannon_label_chisq = np.zeros(num_obj)

for group in np.arange(ngroups):
    inputf = np.load("test_results_%s.npz" %group)
    test_labels = inputf['arr_0']
    nobj = len(test_labels)
    test_errs = inputf['arr_1']
    test_chisqs = inputf['arr_2']
    choose = groups == group
    all_cannon_label_vals[choose] = test_labels
    all_cannon_label_errs[choose] = test_errs
    all_cannon_label_chisq[choose] = test_chisqs

np.savez("xval_cannon_label_vals.npz", all_cannon_label_vals)
np.savez("xval_cannon_label_errs.npz", all_cannon_label_errs)
np.savez("xval_cannon_label_chisq.npz", all_cannon_label_chisq)
