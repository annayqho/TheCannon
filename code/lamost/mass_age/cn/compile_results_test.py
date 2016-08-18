import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import glob
from mass_age_functions import *
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SPEC_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels"
COL_DIR = "/home/annaho/TheCannon/data/lamost"

lab_f = glob.glob("%s/*all_cannon_labels.npz" %COL_DIR)
chisq_f = glob.glob("%s/*cannon_label_chisq.npz" %COL_DIR)
errs_f = glob.glob("%s/*cannon_label_errs.npz" %COL_DIR)
snr_f = glob.glob("%s/*test_snr.npz" %COL_DIR)

id_all = []
teff_all = []
teff_err_all = []
logg_all = []
logg_err_all = []
feh_all = []
feh_err_all = []
cm_all = []
cm_err_all = []
nm_all = []
nm_err_all = []
alpha_all = []
alpha_err_all = []
ak_all = []
ak_err_all = []
snr_all = []
chisq_all = []

for i,f in enumerate(lab_f):
    date = (f.split("/")[-1]).split("_")[0]
    ids = np.load("%s/output/%s_ids.npz" %(SPEC_DIR, date))['arr_0'] 
    labels = np.load(f)['arr_0']
    err = np.load(errs_f[i])['arr_0']
    teff = labels[:,0]
    teff_err = err[:,0]
    logg = labels[:,1]
    logg_err = err[:,1]
    feh = labels[:,2]
    feh_err = err[:,2]
    cm = labels[:,3]
    cm_err = err[:,3]
    nm = labels[:,4]
    nm_err = err[:,4]
    alpha = labels[:,5]
    alpha_err = err[:,5]
    ak = labels[:,6]
    ak_err = err[:,6]
    id_all.extend(ids)
    teff_all.extend(teff)
    teff_err_all.extend(teff_err)
    logg_all.extend(logg)
    logg_err_all.extend(logg_err)
    feh_all.extend(feh)
    feh_err_all.extend(feh_err)
    cm_all.extend(cm)
    cm_err_all.extend(cm_err)
    nm_all.extend(nm)
    nm_err_all.extend(nm_err)
    alpha_all.extend(alpha)
    alpha_err_all.extend(alpha_err)
    ak_all.extend(ak)
    ak_err_all.extend(ak_err)
    chisq_val = np.load(chisq_f[i])['arr_0']
    chisq_all.extend(chisq_val)
    snr_val = np.load(snr_f[i])['arr_0']
    snr_all.extend(snr_val)

teff_all = np.array(teff_all)
logg_all = np.array(logg_all)
feh_all = np.array(feh_all)
cm_all = np.array(cm_all)
nm_all = np.array(nm_all)
alpha_all = np.array(alpha_all)
ak_all = np.array(ak_all)

#mass = calc_mass_2(feh_all, cm_all, nm_all, teff_all, logg_all)
#age = 10.0**calc_logAge(feh_all, cm_all, nm_all, teff_all, logg_all)

np.savez("test_id_all.npz", id_all)
test_label = np.vstack((
    teff_all, logg_all, feh_all, cm_all, nm_all, alpha_all, ak_all))
test_err = np.vstack((
    teff_err_all, logg_err_all, feh_err_all, cm_err_all, nm_err_all, 
    alpha_err_all, ak_err_all))
np.savez("test_label_all.npz", test_label)
np.savez("test_err_all.npz", test_err)
np.savez("teff_all.npz", teff_all)
np.savez("teff_err_all.npz", teff_err_all)
np.savez("logg_all.npz", logg_all)
np.savez("logg_err_all.npz", logg_err_all)
np.savez("feh_all.npz", feh_all)
np.savez("feh_err_all.npz", feh_err_all)
np.savez("cm_all.npz", cm_all)
np.savez("cm_err_all.npz", cm_err_all)
np.savez("nm_all.npz", nm_all)
np.savez("nm_err_all.npz", nm_err_all)
np.savez("alpha_all.npz", alpha_all)
np.savez("alpha_err_all.npz", alpha_err_all)
np.savez("ak_all.npz", ak_all)
np.savez("ak_err_all.npz", ak_err_all)
#np.savez("mass_all.npz", mass)
#np.savez("age_all.npz", age)
np.savez("test_chisq_all.npz", chisq_all)
np.savez("test_snr_all.npz", snr_all)

#tr_feh = np.load("ref_label.npz")['arr_0'][:,2]
#tr_afe = np.load("ref_label.npz")['arr_0'][:,5]

feh_all = np.array(feh_all)
alpha_all = np.array(alpha_all)
#teff_all = np.array(teff_all)
print("%s objects so far" %len(feh_all))
#plt.hist2d(feh_all, alpha_all, norm=LogNorm(), cmap="gray_r", bins=50)
print(feh_all.shape)
plt.hist2d(feh_all, alpha_all, norm=LogNorm(), cmap="gray_r", bins=60, range=[[-2.2,.9],[-.2,.6]])
#choose = teff_all < 4000
#plt.scatter(feh_all, alpha_all, c=teff_all, edgecolor='none', s=1, vmin=3500, vmax=5500)
#plt.scatter(tr_feh, tr_afe, edgecolor='none', c='red', s=1, label="training set")
plt.xlabel("[Fe/H] (dex)" + " from Cannon/LAMOST", fontsize=16)
plt.ylabel(r"$\mathrm{[\alpha/M]}$" + " (dex) from Cannon/LAMOST", fontsize=16)
plt.ylim(-0.2,0.5)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
#plt.legend()
plt.savefig("feh_alpha_temp.png")
