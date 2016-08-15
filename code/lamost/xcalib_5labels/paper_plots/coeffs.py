# Plot the first order coefficients and scatter

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

rc('text', usetex=True)
rc('text.latex', preamble = ','.join('''
\usepackage{txfonts}
\usepackage{lmodern}
'''.split()))
rc('font', family='serif')

direc = "run_9_more_metal_poor"
direc = "run_9b_reddening"

# Load data
#label_names = np.array(['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]'])
label_names = np.array([r'\mathrm{T}_{\mathrm{eff}}', '\mathrm{log g}', 
    '\mathrm{[Fe/H]}', r'[\alphaup/\mathrm{M}]', r'\mathrm{A}_{\mathrm{k}}'])
wl = np.load("../run_2_train_on_good/wl.npz")['arr_0']
coeffs = np.load("../%s/coeffs.npz" %direc)['arr_0']
scatters = np.load("../%s/scatters.npz" %direc)['arr_0']
npixels = len(wl)
nlabels = len(label_names)

# Plot
fig, axarr = plt.subplots(nlabels+1, figsize=(10,10), sharex=True)
ax1 = axarr[0]
plt.subplots_adjust(hspace=0.001)
nbins = len(ax1.get_xticklabels())
for i in range(1,nlabels+1):
    axarr[i].yaxis.set_major_locator(
            MaxNLocator(nbins=nbins, prune='upper'))
plt.xlabel(r"Wavelength $\lambda (\AA)$", fontsize=14)
plt.xlim(np.ma.min(wl), np.ma.max(wl))
#plt.xlim(5760, 5860)
plt.tick_params(axis='x', labelsize=14)
axarr[-1].locator_params(axis='x', nbins=10)

highs = [0.05, 0.016, 0.012, 0.014, 0.3]
lows = [-0.03, -0.018, -0.008, -0.015, -0.45]
errs = [91.5, 0.11, 0.05, 0.05, 1]
vlines = [4066, 4180, 4428, 4502, 4726, 4760, 4763, 4780, 4882, 4964, 5404, 5488, 5494, 5508, 5545, 5705, 5778, 5780, 5797, 5844, 5850, 6010, 6177, 6196, 6203, 6234, 6270, 6284, 
        6376, 6379, 6445, 6533, 6614, 6661, 6699, 
        6887, 6919, 6993, 7224, 7367, 7562, 8621]
vlines = [5173, 5528, 5711, ]

for i in range(0, nlabels):
    d = errs[i]
    ax = axarr[i] 
    lbl = r'$\frac{\partial \mbox{f}}{\partial %s} \delta %s$'%(
            label_names[i], label_names[i])
    ax.set_ylabel(lbl, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.grid(True)
    y = coeffs[:,i+1] * d
    ax.step(wl, y, where='mid', linewidth=0.5, c='k')
    #for line in vlines:
    #    ax.axvline(x=line, c='r')
    ax.locator_params(axis='y', nbins=4)
    ax.set_ylim(lows[i], highs[i])

ax = axarr[nlabels]
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel(r"$s_{\lambda}$", fontsize=14)
top = np.max(scatters[scatters<0.5])
stretch = np.std(scatters[scatters<0.5])
ax.set_ylim(0, top + stretch)
ax.step(wl, scatters, where='mid', c='k', linewidth=0.7)
ax.xaxis.grid(True)
#for line in vlines:
#    ax.axvline(x=line, c='r')
#ax.axvline(x=vline, c='r')
ax.locator_params(axis='y', nbins=4)
fig.savefig('leading_coeffs_5label.png')
#plt.show()
#plt.close(fig)
