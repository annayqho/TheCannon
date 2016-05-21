# Plot the first order coefficients and scatter

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

rc('text', usetex=True)
rc('font', family='serif')

start = 9
end = 13

# Load data
#label_names = ['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]', 'Al', 'Ca',
#        'C', 'Fe', 'K', 'Mg', 'Mn','Na', 'Ni','N', 'O', 'Si', 'S',
#        'Ti', 'V']
label_names = ['T_{eff}', '\log g', 'Al/Fe', 'Ca/Fe',
        'C/Fe', 'Fe/H', 'K/Fe', 'Mg/Fe', 'Mn/Fe','Na/Fe', 'Ni/Fe','N/Fe', 'O/Fe', 'Si/Fe', 
        'S/Fe', 'Ti/Fe', 'V/Fe']
label_names = label_names[start:end]
wl = np.load("run_2_train_on_good/wl.npz")['arr_0']
coeffs = np.load("run_14_all_abundances_fe_xcalib/coeffs.npz")['arr_0']
coeffs = coeffs[:,start:end+1]
scatters = np.load("run_14_all_abundances_fe_xcalib/scatters.npz")['arr_0']
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
#plt.xlim(np.ma.min(wl), 4500)
plt.tick_params(axis='x', labelsize=14)
#axarr[0].set_title(
#        "First-Order Fit Coeffs and Scatter from the Spectral Model",
#        fontsize=14)
axarr[-1].locator_params(axis='x', nbins=10)

#highs = [0.06, 0.02, 0.025, 0.02, 0.017, 
#        0.01, 0.005, 0.005, 0.03, 0.003,
#        0.006, 0.006, 0.005, 0.01, 0.006,
#        0.009, 0.005, 0.004, 0.0025]
highs = [0.06, 0.02, 0.017, 0.01, 0.005, 
        0.005, 0.015, 0.003, 0.006, 
        0.006, 0.0025, 0.005, 0.006,
        0.009, 0.005, 0.004, 0.0025]
#lows = [-0.03, -0.02, -0.025, -0.025, -0.017, 
#        -0.01, -0.004, -0.005, -0.03, -0.003,
#        -0.006, -0.006, -0.005, -0.01, -0.006,
#        -0.01, -0.005, -0.005, -0.002]
lows = [-0.03, -0.02, -0.017, -0.01, -0.004, 
        -0.005, -0.015, -0.003, -0.006, 
        -0.006, -0.0025, -0.01, -0.006,
        -0.01, -0.005, -0.005, -0.002]
#errs = [91.5, 0.11, 0.05, 0.05, 0.07,
#        0.06,0.05,0.05,0.07,0.05,0.06,
#        0.06,0.06,0.05,0.05,0.08,0.06,0.07,0.09]
errs = [91.5, 0.11, 0.07,
        0.06,0.05,0.05,0.07,0.05,0.06,
        0.06,0.06,0.05,0.05,0.08,0.06,0.07,0.09]
highs = highs[start:end]
lows = lows[start:end]
errs = errs[start:end]

for i in range(0, nlabels):
    # scaling factor
    # d = np.sqrt(np.mean((labels[:,i]-np.mean(labels[:,i]))**2)) 
    print(i)
    d = errs[i]
    ax = axarr[i] 
    lbl = r'$\frac{\partial f}{\partial %s} \delta %s$'%(label_names[i], label_names[i])
    ax.set_ylabel(lbl, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.grid(True)
    y = coeffs[:,i+1] * d
    ax.step(wl, y, where='mid', linewidth=0.5, c='k')
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
ax.locator_params(axis='y', nbins=4)
fig.savefig('leading_coeffs_9_13_noMalpha.png')
#plt.show()
plt.close(fig)
