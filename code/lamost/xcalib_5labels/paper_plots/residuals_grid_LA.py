import matplotlib.pyplot as plt
import pyfits
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from math import log10, floor
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

def round_sig(x, sig=2):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)

names = ['\mbox{T}_{\mbox{eff}},', '\mbox{log g}', '\mbox{[Fe/H]}']
units = ['\mbox{K}', '\mbox{dex}', '\mbox{dex}']
snr_str = [r'SNR $\textless$ 50', r'50 $\textless$ SNR $\textless$ 100', r'SNR $\textgreater$ 100']
snr_str = snr_str[::-1]
cutoffs = [0, 50, 100, 10000]
#cutoffs = [0,50,100]
cutoffs = cutoffs[::-1]
y_highs = [300, 0.7, 0.5]
x_lows = [4000, 1.1, -2.0, -0.08]
x_highs = [5300, 3.8, 0.5, 0.4]

direc = "/users/annaho/Data/LAMOST/Label_Transfer"

tr_id = np.load("%s/tr_id.npz" %direc)['arr_0']
tr_id = np.array([val.decode('utf-8') for val in tr_id])
apogee = np.load("%s/tr_label.npz" %direc)['arr_0']
snr = np.load("%s/tr_snr.npz" %direc)['arr_0']

a = pyfits.open("%s/../dr2_stellar.fits" %direc)
dat = a[1].data
a.close()

all_id = dat['lamost_id']
all_lamost_teff = dat['teff']
all_lamost_logg = dat['logg']
all_lamost_feh = dat['feh']

inds = np.array([np.where(all_lamost_lab==val)[0][0] for val in tr_id])
lamost = np.vstack(
        (all_lamost_teff[inds], all_lamost_logg[inds], all_lamost_feh[inds]))

fig = plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(3,3, wspace=0.3, hspace=0)
props = dict(boxstyle='round', facecolor='white', alpha=0.3)

for i in range(0, len(names)):
    name = names[i]
    unit = units[i]

    for j in range(0, len(cutoffs)-1):
        ax = plt.subplot(gs[j,i])
        ax.axhline(y=0, c='k')
        #ax.legend(fontsize=14)
        choose = np.logical_and(snr < cutoffs[j], snr > cutoffs[j+1])
        #choose = snr > cutoffs[j]
        diff = (lamost[:,i] - apogee[:,i])[choose]
        scatter = round_sig(np.std(diff), 3)
        bias  = round_sig(np.mean(diff), 3)

        ax.hist2d(
                apogee[:,i][choose], diff, range=[[x_lows[i], x_highs[i]], [-y_highs[i], y_highs[i]]], bins=30, norm=LogNorm(), cmap="gray_r")

        if j < len(cutoffs) - 2:
            ax.get_xaxis().set_visible(False)
        ax.locator_params(nbins=5)

        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)

        if j == 0:
            ax.set_title(r"$\Delta %s_{\mbox{L-A}}$ [%s]" %(name, unit), fontsize=30)

        if j == 2:
            ax.set_xlabel("$%s$ [%s] from APOGEE" %(name, unit), fontsize=20)

        textstr1 = '%s' %(snr_str[j])
        ax.text(0.05, 0.95, textstr1, transform=ax.transAxes, 
                fontsize=20, verticalalignment='top', bbox=props)
        textstr2 = 'Scatter: %s \nBias: %s' %(scatter, bias)
        ax.text(0.05, 0.05, textstr2, transform=ax.transAxes, 
                fontsize=20, verticalalignment='bottom', bbox=props)
        #ax.set_xlabel(r"APOGEE %s $(%s)$" %(name, unit))
        #ax.set_ylabel(r"Cannon-LAMOST %s $(%s)$" %(name, unit))

plt.savefig("residuals_grid_la.png")
#plt.show()

