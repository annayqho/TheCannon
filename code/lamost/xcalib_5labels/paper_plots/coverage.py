# read in all LAMOST labels

import numpy as np
import pyfits
from matplotlib import rc
from matplotlib import cm
import matplotlib as mpl
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

direc = "/home/annaho/aida41040/annaho/TheCannon/data"

print("reading in all data")
teff = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',', 
        dtype='float', usecols=(1,), skiprows=1)
logg = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',',         
        dtype='float', usecols=(2,), skiprows=1)
feh = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',',
        dtype='float', usecols=(3,), skiprows=1)


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

# plot all

print("plotting")
fig, (ax0,ax1) = plt.subplots(ncols=2, figsize=(12,6), 
                              sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.3)

def dr1(ax):
    ax.hist2d(teff,logg,bins=1000,norm=LogNorm(), cmap="Greys")
    ax.set_ylim(ax0.get_ylim()[1],ax0.get_ylim()[0])
    ax.set_xlim(ax0.get_xlim()[1], ax0.get_xlim()[0])
    ax.set_xlim(7500, 3800)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

dr1(ax0)
dr1(ax1)

cmap = cm.plasma

# plot training set, lamost

lamost_feh[lamost_feh>0.25]=0.25
lamost_feh[lamost_feh<-1.1]=-1.1
im = ax0.scatter(lamost_teff,lamost_logg,c=lamost_feh, s=1, lw=0, cmap=cmap)
cbar = plt.colorbar(im, ax=ax0, label="[Fe/H] [dex] from LAMOST DR2")
cbar.ax.tick_params(labelsize=16)
cbar.set_clim(-1.1,0.25)
ax0.set_xlabel("$\mbox{T}_{\mbox{eff}}$ [K]", fontsize=16)
ax0.set_ylabel("log g [dex]", fontsize=16)
ax0.text(0.05, 0.95, "Colored Points: reference set\nwith their LAMOST labels", 
    horizontalalignment='left', verticalalignment='top', transform=ax0.transAxes,
    fontsize=16)
ax0.text(0.05, 0.80, "Black Points: \n Full LAMOST DR2", transform=ax0.transAxes,
        fontsize=16, verticalalignment='top', horizontalalignment='left')
ax0.locator_params(nbins=5)

# plot training set, apogee

apogee_feh[apogee_feh>0.25] = 0.25
apogee_feh[apogee_feh<-1.1] = -1.1
im = ax1.scatter(apogee_teff,apogee_logg,c=apogee_feh, s=1, lw=0, cmap=cmap)
cbar = plt.colorbar(im, ax=ax1, label="[Fe/H] [dex] from APOGEE DR12")
cbar.ax.tick_params(labelsize=16)
cbar.set_clim(-1.1,0.25)
ax1.set_xlabel("${\mbox{T}_{\mbox{eff}}}$ [K]", fontsize=16)
ax1.set_ylabel("log g [dex]", fontsize=16)
ax1.locator_params(nbins=5)
 
ax1.text(0.05, 0.95, "Colored Points: reference set\nwith their APOGEE labels", 
    horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes,
    fontsize=16)
ax1.text(0.05, 0.80, "Black Points: \n Full LAMOST DR2", transform=ax1.transAxes,
        fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.subplots_adjust(top=0.85)
plt.show()
#plt.savefig("ts_in_full_lamost_label_space.png")
#plt.close()
