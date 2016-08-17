import numpy as np

""" Cull the training set of labels whose errors are > 4 * the scatter """

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/xval_with_cuts_2"

lab = np.load(DATA_DIR + "/xval_cannon_label_vals.npz")['arr_0']
ref_id = np.load(DATA_DIR + "/ref_id.npz")['arr_0']
#all_ref_id = np.load(DATA_DIR + "/ref_id.npz")['arr_0']
#inds = np.array([np.where(all_ref_id==val)[0][0] for val in ref_id])
ref = np.load(DATA_DIR + "/ref_label.npz")['arr_0']
snr = np.load(DATA_DIR + "/ref_snr.npz")['arr_0']

choose = snr > 30
scatters = np.std(ref[choose]-lab[choose], axis=0)
all_bad = np.abs(lab-ref) > 4*scatters[None,:]
bad = np.sum(all_bad, axis=1)
keep = bad == 0

keep_id = ref_id[keep]
np.savez("ref_id_culled.npz", keep_id)
