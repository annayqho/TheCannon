""" Compile the results from the cross-validation """

import numpy as np

ngroups = 8
direc = "/Users/annaho/TheCannon/data/lamost_paper"
ref_id = np.load("%s/ref_id.npz" %direc)['arr_0']
groups = np.load("ref_groups.npz")['arr_0']
ref_label = np.load("%s/ref_label.npz" %direc)['arr_0']
num_obj, num_label = ref_label.shape

all_cannon_label_vals = np.zeros((num_obj, num_label))
all_cannon_label_errs = np.zeros(all_cannon_label_vals.shape)
all_cannon_label_chisq = np.zeros(num_obj)

for group in np.arange(ngroups):
    a = np.load("ex%s_cannon_label_vals.npz" %group)['arr_0']
    b = np.load("ex%s_cannon_label_errs.npz" %group)['arr_0']
    c = np.load("ex%s_cannon_label_chisq.npz" %group)['arr_0']
    choose = groups == group
    all_cannon_label_vals[choose] = a
    all_cannon_label_errs[choose] = b
    all_cannon_label_chisq[choose] = c

np.savez("all_cannon_label_vals.npz", all_cannon_label_vals)
np.savez("all_cannon_label_errs.npz", all_cannon_label_errs)
np.savez("all_cannon_label_chisq.npz", all_cannon_label_chisq)
