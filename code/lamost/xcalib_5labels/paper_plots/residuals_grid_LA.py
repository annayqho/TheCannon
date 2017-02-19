import matplotlib.pyplot as plt
import pyfits
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from math import log10, floor
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
from pylab import rcParams

def round_sig(x, sig=1):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)

names = ['\mbox{T}_{\mbox{eff}},', '\mbox{log g}', '\mbox{[Fe/H]}']
units = ['\mbox{K}', '\mbox{dex}', '\mbox{dex}']
snr_str = r'SNR $\textgreater$ 100'
y_highs = [300, 0.7, 0.5]
x_lows = [4000, 1.1, -2.0, -0.08]
x_highs = [5300, 3.8, 0.5, 0.4]

direc = "/users/annaho/Data/LAMOST/Label_Transfer"

tr_id = np.load("%s/tr_id.npz" %direc)['arr_0']
tr_id = np.array([val.decode('utf-8') for val in tr_id])
apogee = np.load("%s/tr_label.npz" %direc)['arr_0']
snr = np.load("%s/tr_snr.npz" %direc)['arr_0']

a = pyfits.open("%s/lamost_catalog_full.fits" %direc)
dat = a[1].data
a.close()

all_id = dat['lamost_id']
all_lamost_teff = dat['teff_1']
all_lamost_logg = dat['logg_1']
all_lamost_feh = dat['feh']

#print("Finding LAMOST values...")
#inds = np.array([np.where(all_id==val)[0][0] for val in tr_id])
#print("Done finding LAMOST values")
#lamost = np.vstack(
#        (all_lamost_teff[inds], all_lamost_logg[inds], all_lamost_feh[inds]))
#np.savez("tr_label_lamost.npz", lamost)

lamost = np.load("tr_label_lamost.npz")['arr_0'].T

fig,axarr = plt.subplots(1,3)#,figsize=(17,7))
props = dict(boxstyle='round', facecolor='white', alpha=0.3)

for i in range(0, len(names)):
    name = names[i]
    unit = units[i]

    ax = axarr[i]
    ax.axhline(y=0, c='k')
    choose = snr > 100
    print(sum(choose))
    diff = (lamost[:,i] - apogee[:,i])[choose]
    scatter = round_sig(np.std(diff), 1)
    bias  = round_sig(np.mean(diff), 1)

    im = ax.hist2d(
            apogee[:,i][choose], diff, 
            range=[[x_lows[i], x_highs[i]], [-y_highs[i], y_highs[i]]], 
            bins=30, norm=LogNorm(), cmap="gray_r")

    ax.locator_params(nbins=5)

    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)

    ax.set_title(r"$\Delta %s_{\mbox{L-A}}$ [%s]" %(name, unit), fontsize=20)
    ax.set_xlabel("$%s$ [%s] from APOGEE" %(name, unit), fontsize=20)

    textstr1 = '%s' %(snr_str)
    ax.text(0.05, 0.95, textstr1, transform=ax.transAxes, 
            fontsize=20, verticalalignment='top', bbox=props)
    textstr2 = 'Scatter: %s \nBias: %s' %(scatter, bias)
    ax.text(0.05, 0.05, textstr2, transform=ax.transAxes, 
            fontsize=20, verticalalignment='bottom', bbox=props)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = plt.colorbar(im[3], cax=cbar_ax)
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Number of Objects", size=16)
rcParams['figure.figsize'] = 8,3
#plt.tight_layout()
#plt.savefig("residuals_grid_la.png")
plt.show()

