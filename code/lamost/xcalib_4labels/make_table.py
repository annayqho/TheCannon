import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ids_raw = glob.glob("output/*_ids.npz")
ids = []
files = []
chisq = []
for val in ids_raw:
    date = (val.split("/")[1]).split("_")[0]
    fname = "output/%s_all_cannon_labels.npz" %date
    if glob.glob(fname):
        ids.append(val)
        files.append(fname)
        fname_chisq = "output/%s_cannon_label_chisq.npz" %date
        chisq.append(fname_chisq)

id_all = []
teff_all = []
logg_all = []
feh_all = []
alpha_all = []
chisq_all = []

for i,f in enumerate(files):
    id_val = np.load(ids[i])['arr_0']
    id_all.extend(id_val)
    chisq_val = np.load(chisq[i])['arr_0']
    chisq_all.extend(chisq_val)
    labels = np.load(f)['arr_0']
    teff = labels[:,0]
    logg = labels[:,1]
    feh = labels[:,2]
    alpha = labels[:,3]
    teff_all.extend(teff)
    logg_all.extend(logg)
    feh_all.extend(feh)
    alpha_all.extend(alpha)

id_all = np.array(id_all)
teff_all = np.array(teff_all)
logg_all = np.array(logg_all)
feh_all = np.array(feh_all)
alpha_all = np.array(alpha_all)
chisq_all = np.array(chisq_all)
labels = np.vstack((teff_all, logg_all, feh_all, alpha_all))
np.savez("id_all.npz", id_all)
np.savez("label_all.npz", labels)
np.savez("chisq_all", chisq_all)

print("%s objects so far" %len(feh_all))
#plt.hist(chisq_all, bins=500, range=(0,5000))
plt.scatter(feh_all, alpha_all, c=chisq_all, edgecolor='none', s=1, vmin=0, vmax=5000)
plt.colorbar(label="Chi Squared")
choose = chisq_all > 5000
plt.scatter(feh_all[choose], alpha_all[choose], c='r', s=3)
#plt.scatter(tr_feh, tr_afe, edgecolor='none', c='red', s=1, label="training set")
#plt.xlabel("Chi Squared")
#plt.ylabel("Number of Objects")
#plt.title("Distribution of X2")
plt.xlabel(r"$[Fe/H]$" + r" (dex) from Cannon/LAMOST", fontsize=16)
plt.ylabel(r"$[\alpha/Fe]$" + r" (dex) from Cannon/LAMOST", fontsize=16)
#plt.ylim(-0.2,0.5)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
#plt.legend()
#plt.savefig("feh_alpha_cX2.png")
#plt.show()
