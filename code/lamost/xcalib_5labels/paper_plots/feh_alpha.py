import numpy as np
import matplotlib.pyplot as plt
import glob
import pyfits
from corner import hist2d
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
#rc('text.latex', preamble = ','.join('''
#   \usepackage{txfonts}
#   \usepackage{lmodern}
#   '''.split()))
plt.rc('font', family='serif')

direc = '/users/annaho/Data/LAMOST/Label_Transfer'
f = pyfits.open("%s/table_for_paper.fits" %direc)
a = f[1].data
f.close()

feh = a['cannon_m_h']
am = a['cannon_alpha_m']
snr = a['snrg']

choose = snr > 20

fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(8,5))
hist2d(feh[choose], am[choose], ax=ax, bins=100, range=[[-2.2,.9],[-0.2,0.5]])
ax.set_xlabel("[Fe/H] (dex)" + " from Cannon/LAMOST", fontsize=16)
fig.text(
        0.04, 0.5, r"$\mathrm{[\alpha/M]}$" + " (dex) from Cannon/LAMOST", 
        fontsize=16, va = 'center', rotation='vertical')
label = r"Objects with SNR \textgreater 20"
props = dict(boxstyle='round', facecolor='white')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.text(0.05, 0.85, label, 
        horizontalalignment='left', verticalalignment='bottom', 
        transform=ax.transAxes, fontsize=16, bbox=props)
#plt.show()
plt.savefig("feh_alpha.png")
