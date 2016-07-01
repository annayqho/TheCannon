import pyfits
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np


direc = "/Users/annaho/TheCannon/data/lamost_paper"
snr = np.load("%s/ref_snr.npz" %direc)['arr_0']
apogee = np.load("%s/ref_label.npz" %direc)['arr_0']
cannon = np.load("../all_cannon_label_vals.npz")['arr_0']

hdulist = pyfits.open("%s/lamost_catalog_training.fits" %direc)
tbdata = hdulist[1].data
hdulist.close()
snrg = tbdata.field("snrg")
snri = tbdata.field("snri")
lamost_id_full = tbdata.field("lamost_id")
lamost_id = np.array([val.strip() for val in lamost_id_full])
lamost_teff = tbdata.field("teff_1")
lamost_logg = tbdata.field("logg_1")
lamost_feh = tbdata.field("feh")
lamost = np.vstack((lamost_teff, lamost_logg, lamost_feh)).T

data = [lamost, cannon, apogee]

low = 3800
high = 5500

low2 = 0.5
high2 = 4.0

fig,axarr = plt.subplots(1,3, figsize=(10,5.5), sharex=True, sharey=True)

names = ['LAMOST DR2', 'Cannon/LAMOST', 'APOGEE DR12']

for i in range(0, len(names)):
    ax = axarr[i]
    use = data[i]
    im = ax.hist2d(use[:,0], use[:,1], norm=LogNorm(), bins=100, 
            cmap="inferno", range=[[low,high],[low2,high2]], vmin=1,vmax=70)
    ax.set_xlabel(r"$\mbox{T}_{\mbox{eff}}$" + " [K]", fontsize=16)
    if i == 0:
        ax.set_ylabel("log g [dex]", fontsize=16)
    ax.set_title("%s" %names[i], fontsize=16)
    ax.set_xlim(low,high)
    ax.set_ylim(low2,high2)
    ax.tick_params(axis='x', labelsize=16)
    ax.locator_params(nbins=5)
    #if i == 2: fig.colorbar(im[3], cax=ax, label="log(Number of Objects)")
    #plt.savefig("rc_%s.png" %names)
    #plt.close()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = plt.colorbar(im[3], cax=cbar_ax)
cbar.set_label("log(density)", size=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(labelsize=16)

plt.show()
#plt.savefig("rc_5label.png")
