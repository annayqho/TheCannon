# Plot the first order coefficients and scatter

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

rc('text', usetex=True)
rc('font', family='serif')

label_names = ['T_{eff}', '\log g', 'Al/Fe', 'Ca/Fe',
        'C/Fe', 'Fe/H', 'K/Fe', 'Mg/Fe', 'Mn/Fe','Na/Fe', 'Ni/Fe','N/Fe', 'O/Fe', 'Si/Fe', 
        'S/Fe', 'Ti/Fe', 'V/Fe']
wl = np.load("run_2_train_on_good/wl.npz")['arr_0']
coeffs = np.load("run_14_all_abundances_fe_xcalib/coeffs.npz")['arr_0']
scatters = np.load("run_14_all_abundances_fe_xcalib/scatters.npz")['arr_0']
npixels = len(wl)
nlabels = len(label_names)

# C, N, O, Na, Mg, Si, Ca, Ti, Fe

choose = [4,11,12,9,7,13,3,15,5]
choose = [4,11,12,9]

# plot
fig = plt.figure(figsize=(13,12))
gs = gridspec.GridSpec(2,2,wspace=0.3,hspace=0.3)

highs = np.array([0.06, 0.02, 0.017, 0.01, 0.005,
        0.005, 0.015, 0.003, 0.006, 
        0.006, 0.0025, 0.005, 0.006,
        0.009, 0.005, 0.004, 0.0025])
#highs[choose[2]] = 0.007
#highs[choose[3]] = 0.005
lows = np.array([-0.03, -0.02, -0.017, -0.01, -0.004, 
        -0.005, -0.015, -0.003, -0.006, 
        -0.006, -0.0025, -0.01, -0.006,
        -0.01, -0.005, -0.005, -0.002])
#lows[choose[1]] = -.007
#lows[choose[2]] = -0.008
#lows[choose[3]] = -0.008
#lows[choose[4]] = -0.004
errs = [91.5, 0.11, 0.07,
        0.06,0.05,0.05,0.07,0.05,0.06,
        0.06,0.06,0.05,0.05,0.08,0.06,0.07,0.09]

for i,ind in enumerate(choose):
    ax = plt.subplot(gs[i])
    ax.set_xlabel(r"$\lambda (\AA)$", fontsize=12)
    ax.set_xlim(4000,6500)
    plt.tick_params(axis='x', labelsize=14)

    d = errs[ind]
    lbl = r'$\frac{\partial f}{\partial %s} \delta %s$'%(label_names[ind], label_names[ind])
    #ax.set_ylabel(lbl, fontsize=14)
    ax.text(0.85,0.90,label_names[ind],transform=ax.transAxes,horizontalalignment='right')
    plt.tick_params(axis='y', labelsize=14)
    ax.xaxis.grid(True)
    y = coeffs[:,ind+1] * d
    ax.step(wl, y, where='mid', linewidth=0.5, c='k')
    ax.locator_params(axis='y', nbins=4)
    ax.set_ylim(lows[ind], highs[ind])

plt.show()
