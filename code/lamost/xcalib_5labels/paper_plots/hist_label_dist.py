import numpy as np
import pyfits
from matplotlib import rc
from matplotlib import cm
import matplotlib as mpl
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


print("reading in apogee labels")
# read in apogee labels

tr_IDs = np.load("../tr_id.npz")['arr_0']
labels_apogee = np.load("../tr_label.npz")['arr_0']
apogee_teff = labels_apogee[:,0]
apogee_logg = labels_apogee[:,1]
apogee_feh = labels_apogee[:,2]

# read in lamost labels

print("reading in lamost labels")
a = pyfits.open("../../make_lamost_catalog/lamost_catalog_training.fits")
b = a[1].data
a.close()
IDs_lamost = b['lamost_id']
IDs_lamost = np.array([val.strip() for val in IDs_lamost])
teff_all_lamost = b['teff_1']
logg_all_lamost = b['logg_1']
feh_all_lamost = b['feh']
inds = np.array([np.where(IDs_lamost==a)[0][0] for a in tr_IDs])
lamost_teff = teff_all_lamost[inds]
lamost_logg = logg_all_lamost[inds]
lamost_feh = feh_all_lamost[inds]

# histogram of LAMOST values and APOGEE values for Teff, logg, Fe/H

fig, (ax0,ax1,ax2) = plt.subplots(ncols=3, figsize=(10,4),
                              sharex=False, sharey=False)
plt.subplots_adjust(wspace=0.3)
bins = 50
col_apogee = 'darkorchid'
col_lamost = 'darkorange'
ax0.hist(
    lamost_teff, color=col_lamost, histtype='step', 
    bins=bins, range=(3500,6000), normed=True, label="LAMOST")
ax0.hist(
    apogee_teff, color=col_apogee, histtype='step', 
    bins=bins, range=(3500,6000), normed=True, label="APOGEE")
ax0.set_xlabel("${\mbox{T}_{\mbox{eff}}}$ [K]", fontsize=14)
ax0.legend(loc="upper left")
ax0.set_ylabel("Num. Training Objects, Normalized", fontsize=14)
ax0.set_ylim(0,0.003)
ax0.locator_params(nbins=5)
ax0.tick_params(axis='both', labelsize=14)
ax1.hist(
    lamost_logg, color=col_lamost, histtype='step', 
    bins=bins, range=(0.2,4.8), normed=True, label="LAMOST")
ax1.hist(
    apogee_logg, color=col_apogee, histtype='step', 
    bins=bins, range=(0.2,4.8), normed=True, label="APOGEE")
ax1.legend(loc="upper left")
ax1.set_ylim(0,2)
ax1.locator_params(nbins=5)
ax1.set_xlabel("log g [dex]", fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax2.hist(
    lamost_feh, color=col_lamost, histtype='step', 
    bins=bins, normed=True, range=(-2,0.5), label="LAMOST")
ax2.hist(
    apogee_feh, color=col_apogee, histtype='step', 
    bins=bins, normed=True, range=(-2, 0.5), label="APOGEE")
ax2.legend(loc="upper left")
ax2.set_xlabel("[Fe/H] [dex]")
ax2.locator_params(nbins=5)
ax2.set_ylim(0,2)
ax2.tick_params(axis='both', labelsize=14)

#plt.show()
plt.tight_layout()
plt.savefig("hist_label_dist.png")
plt.close()
