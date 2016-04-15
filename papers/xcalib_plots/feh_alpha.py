import numpy as np
import matplotlib.pyplot as plt
import glob
import pyfits
from corner import hist2d
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
rc('text.latex', preamble = ','.join('''
   \usepackage{txfonts}
   \usepackage{lmodern}
   '''.split()))
plt.rc('font', family='serif')

f = pyfits.open("../make_lamost_catalog/table_for_paper.fits")
a = f[1].data
f.close()

feh = a['cannon_m_h']
am = a['cannon_alpha_m']
snr = a['snrg']

choose = snr > 50

fig, axarr = plt.subplots(2,1, sharex=True, sharey=True, figsize=(8,10))
hist2d(feh[choose], am[choose], ax=axarr[0], bins=100, range=[[-2.2,.9],[-0.2,0.5]])
hist2d(feh, am, ax=axarr[1], bins=100, range=[[-2.2,.9],[-0.2,0.5]])
axarr[1].set_xlabel("[Fe/H] (dex)" + " from Cannon/LAMOST", fontsize=16)
#axarr[1].set_ylabel(r"$\mathrm{[\alphaup/M]}$" + " (dex) from Cannon/LAMOST", fontsize=16)
fig.text(0.04, 0.5, r"$\mathrm{[\alphaup/M]}$" + " (dex) from Cannon/LAMOST", fontsize=16,
        va = 'center', rotation='vertical')
labels = [r"Objects with SNR \textgreater 50", r"All Objects"]
props = dict(boxstyle='round', facecolor='white')
for i,ax in enumerate(axarr):
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.text(0.05, 0.85, labels[i], 
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes,
            fontsize=16, bbox=props)
#plt.show()
plt.savefig("feh_alpha.png")
