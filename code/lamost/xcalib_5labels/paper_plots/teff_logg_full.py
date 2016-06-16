import pyfits
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

direc = "/Users/annaho/TheCannon/code/lamost/make_lamost_catalog"
hdulist = pyfits.open("%s/lamost_catalog_full.fits" %direc)
tbdata = hdulist[1].data
hdulist.close()
teff = tbdata.field("cannon_teff")
logg = tbdata.field("cannon_logg")
mh = tbdata.field("cannon_m_h")
snr = tbdata.field("cannon_snrg")
cannon = np.vstack((teff, logg, mh)).T

teff = tbdata.field("teff_1")
logg = tbdata.field("logg_1")
feh = tbdata.field("feh")
lamost = np.vstack((teff, logg, mh)).T

choose = np.logical_and(feh < 0.0, feh > -0.1)
print(sum(choose))

data = [lamost[choose], cannon[choose]]

low = 3600
high = 6000

low2 = 0.0
high2 = 4.5 

fig,axarr = plt.subplots(1,2, figsize=(10,5.5), sharex=True, sharey=True)

names = ['Labels from LAMOST DR2', 'Labels from Cannon']

#text = r'-0.1 $\textless$ [Fe/H] $\textless$ 0.0 (44,000 objects)'

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

#props = dict(boxstyle='round', facecolor='white')
# axarr[0].text(
#         0.5, 0.90, text, horizontalalignment='left', 
#         verticalalignment='top', transform=axarr[0].transAxes, bbox=props,
#         fontsize=16)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = plt.colorbar(im[3], cax=cbar_ax)
cbar.set_label("log(density)", size=16)
cbar.ax.tick_params(labelsize=16)

#plt.show()
plt.savefig("teff_logg_test_set.png")
