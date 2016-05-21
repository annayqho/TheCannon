import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ids_raw = glob.glob("../xcalib_4labels/output/*_ids.npz")
ids = []
files = []
err_files = []
npix_files = []
chisq = []
for val in ids_raw:
    date = (val.split("/")[3]).split("_")[0]
    print(date)
    fname = "output/%s_all_cannon_labels.npz" %date
    if glob.glob(fname):
        ids.append(val)
        files.append(fname)
        errfname = "output/%s_cannon_label_errs.npz" %date
        err_files.append(errfname)
        npixfname = "../xcalib_4labels/output/%s_npix.npz" %date
        npix_files.append(npixfname)
        fname_chisq = "output/%s_cannon_label_chisq.npz" %date
        chisq.append(fname_chisq)

id_all = []
teff_all = []
teff_err_all = []
logg_all = []
logg_err_all = []
feh_all = []
feh_err_all = []
alpha_all = []
alpha_err_all = []
ak_all = []
ak_err_all = []
npix_all = []
chisq_all = []

for i,f in enumerate(files):
    id_val = np.load(ids[i])['arr_0']
    id_all.extend(id_val)
    chisq_val = np.load(chisq[i])['arr_0']
    chisq_all.extend(chisq_val)
    npix_val = np.load(npix_files[i])['arr_0']
    npix_all.extend(npix_val)
    labels = np.load(f)['arr_0']
    errs = np.load(err_files[i])['arr_0']
    teff = labels[:,0]
    teff_err = errs[:,0]
    logg = labels[:,1]
    logg_err = errs[:,1]
    feh = labels[:,2]
    feh_err = errs[:,2]
    alpha = labels[:,3]
    alpha_err = errs[:,3]
    ak = labels[:,4]
    ak_err = errs[:,4]
    teff_all.extend(teff)
    teff_err_all.extend(teff_err)
    logg_all.extend(logg)
    logg_err_all.extend(logg_err)
    feh_all.extend(feh)
    feh_err_all.extend(feh_err)
    alpha_all.extend(alpha)
    alpha_err_all.extend(alpha_err)
    ak_all.extend(ak)
    ak_err_all.extend(ak_err)

id_all = np.array(id_all)
teff_all = np.array(teff_all)
teff_err_all = np.array(teff_err_all)
logg_all = np.array(logg_all)
logg_err_all = np.array(logg_err_all)
feh_all = np.array(feh_all)
feh_err_all = np.array(feh_err_all)
alpha_all = np.array(alpha_all)
alpha_err_all = np.array(alpha_err_all)
ak_all = np.array(ak_all)
ak_all_err = np.array(ak_err_all)
npix_all = np.array(npix_all)
chisq_all = np.array(chisq_all)
labels = np.vstack((teff_all, logg_all, feh_all, alpha_all, ak_all))
errs = np.vstack((teff_err_all, logg_err_all, feh_err_all, alpha_err_all, ak_err_all))
np.savez("id_all.npz", id_all)
np.savez("label_all.npz", labels)
np.savez("npix_all.npz", npix_all)
np.savez("chisq_all.npz", chisq_all)
np.savez("errs_all.npz", errs)

print("%s objects so far" %len(feh_all))
#plt.hist(chisq_all, bins=500, range=(0,5000))
plt.scatter(feh_all, alpha_all, c=chisq_all, edgecolor='none', s=1, vmin=0, vmax=5000)
plt.colorbar(label="Chi Squared")
#choose = chisq_all > 0
choose = chisq_all > 5000
plt.scatter(feh_all[choose], alpha_all[choose], c='r', s=3)
#plt.scatter(tr_feh, tr_afe, edgecolor='none', c='red', s=1, label="training set")
#plt.xlabel("Chi Squared")
#plt.ylabel("Number of Objects")
#plt.title("Distribution of X2")
plt.xlabel(r"$[Fe/H]$" + r" (dex) from Cannon/LAMOST", fontsize=16)
plt.ylabel(r"$[\alpha/Fe]$" + r" (dex) from Cannon/LAMOST", fontsize=16)
plt.ylim(-0.2,0.5)
plt.xlim(-2.0, 1)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
#plt.legend()
plt.savefig("feh_alpha_cX2.png")
#plt.show()
