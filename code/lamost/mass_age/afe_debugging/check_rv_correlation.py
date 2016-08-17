import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append("/Users/annaho/Dropbox/Research/TheCannon/code/lamost/mass_age")
from mass_age_functions import *
from marie_cuts import get_mask

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask"

ids = np.load("%s/ref_id_col.npz" %DATA_DIR)['arr_0']
labels = np.load(DATA_DIR + "/xval_one_iteration/xval_cannon_label_vals.npz")['arr_0']
chisq = np.load(DATA_DIR + "/xval_one_iteration/xval_cannon_label_chisq.npz")['arr_0']
snr = np.load(DATA_DIR + "/ref_snr.npz")['arr_0']
#ref_label = np.load(DATA_DIR + "/xval_one_iteration/ref_label.npz")['arr_0']
ref_label = np.load(DATA_DIR + "/ref_label.npz")['arr_0']

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"

ref_ids = np.load(DATA_DIR + "/no_wl_cuts/ref_id.npz")['arr_0']
inds = np.array([np.where(ref_ids==val)[0][0] for val in ids])
rv = np.load(DATA_DIR + "/no_wl_cuts/ref_rvs.npz")['arr_0'][inds]

choose_quality = np.logical_and(snr > 20, chisq < 9000)
choose_teff = teff > 4250
choose = np.logical_and(choose_teff, choose_quality)
#choose = snr > 70
#choose = snr > 0
diff = labels - ref_label
diff = labels[:,5]-ref_label[:,5]

plt.scatter(diff[choose], rv[choose], lw=0, c='k')
#plt.show()

teff = labels[:,0]
logg = labels[:,1]
mh = labels[:,2]
cm = labels[:,3]
nm = labels[:,4]
afe = labels[:,5]
mask = get_mask(
        labels[:,0], labels[:,1], labels[:,2], 
        labels[:,3], labels[:,4], labels[:,5])
mass = calc_mass_2(mh, cm, nm, teff, logg)
age = calc_logAge(mh, cm, nm, teff, logg)
ref_age = calc_logAge(
        ref_label[:,2], ref_label[:,3], ref_label[:,4], 
        ref_label[:,0], ref_label[:,1])

#choose_age = np.logical_and(age < 12, age > 0)
#keep_a = np.logical_and(choose, mask)
#keep = np.logical_and(keep_a, choose_age)
keep = np.logical_and(choose, mask)

plt.scatter(
       labels[:,2][keep], labels[:,5][keep], c=age[keep],
        s=5, lw=0, cmap=cm.RdYlBu_r, vmin=0, vmax=12)
# 
# plt.scatter(
#         ref_label[:,5][choose], labels[:,5][choose], c=rv[choose], 
j         s=5, lw=0, cmap=cm.RdYlBu, vmin=-200, vmax=120)
# plt.show()
