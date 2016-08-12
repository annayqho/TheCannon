import pyfits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

direc = "/Users/annaho/Data/LAMOST/Mass_And_Age"
hdulist = pyfits.open("%s/lamost_catalog_mass_age_with_cuts.fits" %direc)
tbdata = hdulist[1].data
hdulist.close()
mh = tbdata.field("cannon_mh")
afe = tbdata.field("cannon_afe")
age = tbdata.field("cannon_age")
snr = tbdata.field("cannon_snrg")
#cannon = np.vstack((cannon_mh, cannon_afe, cannon_age)).T

low = min(mh)
high = max(mh)

low2 = -0.1
high2 = 0.35

fig,axarr = plt.subplots(1,3, figsize=(15,5.5), sharex=True, sharey=True)

names = [r'SNR \textgreater 100', r'100 \textgreater SNR \textgreater 60', 
r'60 \textgreater SNR \textgreater 30']
snr_min = [100, 60, 30]
cmap = cm.inferno

for i in range(0, len(names)):
    ax = axarr[i]
    if i == 0:
        choose = snr > 100
    elif i == 1:
        choose = np.logical_and(snr > 60, snr < 100)
    elif i == 2:
        choose = np.logical_and(snr < 60, snr > 30)
    im = ax.scatter(
            mh[choose],afe[choose],c=age[choose], 
            s=1, lw=0, cmap=cmap, vmin=1, vmax=12,
            norm=LogNorm())
    ax.set_xlabel(r"$\mbox{[Fe/H] [K]}$", fontsize=16)
    if i == 0:
        ax.set_ylabel(r"$\mbox{[\\alpha/Fe] [dex]}$", fontsize=16)
    ax.set_title("%s" %names[i], fontsize=16)
    ax.set_xlim(low,high)
    ax.set_ylim(low2,high2)
    ax.tick_params(axis='x', labelsize=16)
    ax.locator_params(nbins=5)
    #if i == 2: fig.colorbar(im[3], cax=ax, label="log(Number of Objects)")
    #plt.savefig("rc_%s.png" %names)
    #plt.close()

#props = dict(boxstyle='round', facecolor='white')
# axarr[0].text(
#         0.5, 0.90, text, horizontalalignment='left', 
#         verticalalignment='top', transform=axarr[0].transAxes, bbox=props,
#         fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = plt.colorbar(im, cax=cbar_ax)
#cbar.set_clim(0,12)
cbar.set_label("Age [Gyr]", size=16)
cbar.ax.tick_params(labelsize=16)

plt.show()
#plt.savefig("teff_logg_test_set.png")
