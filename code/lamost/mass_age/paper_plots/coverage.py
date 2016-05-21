# read in all LAMOST labels

import numpy as np
from matplotlib import rc
from matplotlib import cm
import matplotlib as mpl
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

direc = "/home/annaho/aida41040/annaho/TheCannon/examples"

teff = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',', 
        dtype='float', usecols=(1,), skiprows=1)
logg = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',',         
        dtype='float', usecols=(2,), skiprows=1)
feh = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',',
        dtype='float', usecols=(3,), skiprows=1)

# read in cannon labels

#labels_cannon = np.load("%s/test_training_overlap/test_labels.npz" %direc)['arr_0']
labels_cannon = np.load("../run_9_more_metal_poor/all_cannon_labels.npz")['arr_0']
cannon_teff = labels_cannon[:,0]
cannon_logg = labels_cannon[:,1]
cannon_feh = labels_cannon[:,2]

# read in apogee labels

direc_apogee = "../run_9_more_metal_poor/" 
tr_IDs = np.load("%s/tr_id.npz" %direc_apogee)['arr_0']
labels_apogee = np.load("%s/tr_label.npz" %direc_apogee)['arr_0']
apogee_teff = labels_apogee[:,0]
apogee_logg = labels_apogee[:,1]
apogee_feh = labels_apogee[:,2]

# read in lamost labels

IDs_lamost = np.loadtxt(
            "%s/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt" %direc, 
                usecols=(0,), dtype=(str))
labels_all_lamost = np.loadtxt(
            "%s/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt" %direc, 
                usecols=(3,4,5), dtype=(float))
inds = np.array([np.where(IDs_lamost==a)[0][0] for a in tr_IDs])
labels_lamost = labels_all_lamost[inds,:]
lamost_teff = labels_lamost[:,0]
lamost_logg = labels_lamost[:,1]
lamost_feh = labels_lamost[:,2]

# plot all

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
#plt.show()
plt.savefig("ts_in_full_lamost_label_space.png")
plt.close()
