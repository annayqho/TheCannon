import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)

# the cannon-apogee label, trained on kepler labels
cannon_label = np.load("test_labels.npz")['arr_0']

# the ids of the apogee stars whose labels these are
cannon_ids = np.load("test_ids.npz")['arr_0']
cannon_ids = np.array([a.split('/')[-1] for a in cannon_ids])

# the ids of the apogee-lamost training objects
apogee_lamost_tr_ids = np.loadtxt(
    "../../PAPER_training_step/tr_files_apogee.txt", dtype=str)

# get corresponding DR12 labels for the apogee-lamost objects
IDs_apogee = np.loadtxt(
        "../../apogee_dr12_labels.csv", usecols=(1,),
        delimiter=',', dtype=(str))
labels_all_apogee = np.loadtxt(
        "../../apogee_dr12_labels.csv", usecols=(2,3,4,5),
        delimiter=',', dtype=(float))
inds = np.array([np.where(IDs_apogee==a)[0][0] for a in apogee_lamost_tr_ids])
apogee_label = labels_all_apogee[inds,:]
# apogee_label = labels_all_apogee

# pick corresponding cannon-apogee labels for the apogee-lamost objs
inds = np.array([np.where(cannon_ids==a)[0][0] for a in apogee_lamost_tr_ids])
cannon_label = cannon_label[inds,:]

SNR = np.load("test_SNR.npz")['arr_0']
SNR = SNR[inds]

cmap = 'cool'

vmin = 50
vmax = 200

fig, axarr = subplots(2,2)
ax = axarr[0,0]
ax.scatter(apogee_label[:,0], cannon_label[:,0], marker='x', c=SNR, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
#ax.scatter(apogee_label[:,0], cannon_label[:,0], marker='x', c=chisq, cmap=cmap, alpha=0.5, vmin=20, vmax=100)
ax.set_xlabel(r"APOGEE DR12 $T_{eff}$")
ax.set_ylabel(r"Cannon-APOGEE (Kepler) $T_{eff}$")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.plot(ylim,ylim, c='k')
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax = axarr[0,1]
ax.scatter(apogee_label[:,1], cannon_label[:,1], marker='x', c=SNR, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
ax.set_xlabel(r"APOGEE DR12 $\log g$")
ax.set_ylabel(r"Cannon-APOGEE (Kepler) $\log g$")
xlim = ax.get_xlim()
ylim = ax.get_ylim() 
ax.plot(ylim,ylim, c='k')
ax.set_xlim(xlim)
ax.set_ylim(ylim) 

ax = axarr[1,0]
ax.scatter(apogee_label[:,2], cannon_label[:,2], marker='x', c=SNR, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
ax.set_xlabel(r"APOGEE DR12 $[Fe/H]$")
ax.set_ylabel(r"Cannon-APOGEE (Kepler) $[Fe/H]$")
xlim = ax.get_xlim()
ylim = ax.get_ylim() 
ax.plot(ylim,ylim, c='k')
ax.set_xlim(xlim)
ax.set_ylim(ylim) 

ax = axarr[1,1]
im = ax.scatter(apogee_label[:,3], cannon_label[:,3], marker='x', c=SNR, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
ax.set_xlabel(r"APOGEE DR12 $[\alpha/Fe]$")
ax.set_ylabel(r"Cannon-APOGEE (Kepler) $[\alpha/Fe]$")
xlim = ax.get_xlim()
ylim = ax.get_ylim() 
ax.plot(ylim,ylim, c='k')
ax.set_xlim(xlim)
ax.set_ylim(ylim) 

subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
colorbar(im, cax=cbar_ax)
