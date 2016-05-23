import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)

# the cannon-apogee label, trained on kepler labels
cannon_label = np.load("test_label.npz")['arr_0']

# get corresponding DR12 labels 
apogee_label = np.load("../tr_label.npz")['arr_0'] 

SNR = np.load("test_SNR.npz")['arr_0']

cmap = 'cool'

vmin = 60 
vmax = 100 

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
