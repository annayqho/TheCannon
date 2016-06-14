import numpy as np
import os

# APOGEE-APOKASC overlap

inputf = "/home/annaho/TheCannon/examples/example_apokasc/apokasc_DR12_overlap.npz"
apogee_apokasc = np.load(inputf)['arr_0']

# APOGEE-LAMOST overlap

inputf = "/home/annaho/TheCannon/examples/example_DR12/Data"
apogee_lamost = np.array(os.listdir(inputf))

# APOGEE-APOKASC-LAMOST

overlap = np.intersect1d(apogee_lamost, apogee_apokasc) # 530 stars
apogee_key = np.loadtxt("apogee_sorted_by_ra.txt", dtype=str)
lamost_key = np.loadtxt("lamost_sorted_by_ra.txt", dtype=str)
inds = np.array([np.where(apogee_key==a)[0][0] for a in overlap])
overlap_lamost = lamost_key[inds]

np.savez("apogee_apokasc_lamost_overlap.npz", overlap)

# get all APOGEE parameters

label_file = "apogee_dr12_labels.csv"
apogee_id_all = np.loadtxt(label_file, usecols=(1,), delimiter=',', dtype=str)
apogee_labels_all = np.loadtxt(
        label_file, usecols=(2,3,4,5), delimiter=',', dtype=float)
inds = np.array([np.where(apogee_id_all==a)[0][0] for a in overlap])
apogee_id = apogee_id_all[inds]
apogee_labels = apogee_labels_all[inds,:]


# get all APOKASC parameters

apokasc_id_all = np.load("example_apokasc/apokasc_DR12_overlap.npz")['arr_0']
apokasc_labels_all = np.load("example_apokasc/tr_label.npz")['arr_0']
inds = np.array([np.where(apokasc_id_all==a)[0][0] for a in overlap])
apokasc_id = apokasc_id_all[inds]
apokasc_labels = apokasc_labels_all[inds]


# get all LAMOST parameters

inputf = "/home/annaho/TheCannon/examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt"
lamost_id_all = np.loadtxt(inputf, usecols=(0,), dtype=str)
lamost_labels_all = np.loadtxt(inputf, usecols=(3,4,5), dtype=float)
inds = np.array([np.where(lamost_id_all==a)[0][0] for a in overlap_lamost])
lamost_id = lamost_id_all[inds]
lamost_labels = lamost_labels_all[inds]

# plot them against each other
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
names = [r"$T_{eff}$", r"$\log g$", r"$[Fe/H]$", r"$[\alpha/Fe]$"]

def plot(ax, x, y, i):
    ax.scatter(x[:,i], y[:,i], c='k')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([-10000,10000],[-10000,10000], c='r')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(names[i])

x = apokasc_labels
y = lamost_labels
fig,axarr = subplots(2,2)

ax = axarr[0,0]
plot(ax, x, y, 0)
ax = axarr[0,1]
plot(ax, x, y, 1)
ax = axarr[1,0]
plot(ax, x, y, 2)
#ax = axarr[1,1]
#plot(ax, x, y, 3)
fig.text(0.5,0.01, "Kepler APOKASC", ha='center', va='bottom', fontsize=18)
fig.text(0.01, 0.5, "LAMOST", ha='left', va='center', rotation=90, fontsize=18)
