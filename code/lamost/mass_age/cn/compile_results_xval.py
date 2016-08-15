""" Compile the results from the cross-validation """

import numpy as np

ngroups = 8
direc_old = "/Users/annaho/Data/LAMOST/Mass_And_Age"
direc = "."

ref_id = np.load("%s/ref_id.npz" %direc_old)['arr_0']
groups = np.load("%s/assignments.npz"%direc_old)['arr_0']

real_ref_ids = np.load("./ref_id_col.npz")['arr_0']
inds = np.array([np.where(ref_id==val)[0][0] for val in real_ref_ids])
ref_id = ref_id[inds]
groups = groups[inds]

#ref_label = np.load("%s/ref_label.npz" %direc_old)['arr_0']
ref_label = np.load("%s/ref_label.npz" %direc_old)['arr_0'][inds]
ref_snr = np.load("%s/ref_snr.npz" %direc_old)['arr_0'][inds]
# CHANGE FOR NO AK
#ref_label = np.load("%s/ref_label.npz" %direc_old)['arr_0'][:,0:6]
num_obj, num_label = ref_label.shape

tr_label_vals = np.zeros((num_obj, num_label))
all_ref_snr = np.zeros(num_obj)
all_cannon_label_vals = np.zeros((num_obj, num_label))
all_cannon_label_errs = np.zeros(all_cannon_label_vals.shape)
all_cannon_label_chisq = np.zeros(num_obj)

for group in np.arange(ngroups):
    inputf = np.load("all_colors_culled_test_results_%s.npz" %group)
    test_labels = inputf['arr_0']
    nobj = len(test_labels)
    test_errs = inputf['arr_1']
    test_chisqs = inputf['arr_2']
    choose = groups == group
    all_cannon_label_vals[choose] = test_labels
    all_cannon_label_errs[choose] = test_errs
    all_cannon_label_chisq[choose] = test_chisqs
    all_ref_snr[choose] = ref_snr[choose]

np.savez("all_colors_culled_xval_cannon_label_vals.npz", all_cannon_label_vals)
np.savez("all_colors_culled_xval_cannon_label_errs.npz", all_cannon_label_errs)
np.savez("all_colors_culled_xval_cannon_label_chisq.npz", all_cannon_label_chisq)
np.savez("all_colors_culled_xval_ref_snr.npz", all_ref_snr)
