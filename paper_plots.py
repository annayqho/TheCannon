import pickle
import numpy as np
import matplotlib.pyplot as plt
from cannon.dataset import Dataset
from cannon.helpers import Table
from matplotlib import rc
import matplotlib.gridspec as gridspec
import pickle

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

test_SNR = pickle.load(open('test_SNR.p', 'r'))
test_labels = pickle.load(open('test_labels.p', 'r'))
label_file = "example_DR12/apogee_test_labels.csv"
data = Table(label_file)
data.sort('id')
label_names = data.keys()
nlabels = len(label_names)
apogee_label_vals = np.array([data[k] for k in label_names]).T

names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]'] 

for i in range(0, len(names)):
    name = names[i]
    cannon = np.array(test_labels[:,i])
    orig = np.array(apogee_label_vals[:,i+1], dtype=float)
    snr = test_SNR
    name = names[i]
    bad = orig < -8000
    good = snr > 50
    orig = np.ma.array(orig, mask=bad)
    cannon = np.ma.array(cannon, mask=bad)
    snr = np.ma.array(snr, mask=bad)
    orig = orig[good]
    cannon = cannon[good]
    snr = snr[good]

    scatter = np.round(np.std(orig-cannon),3)
    bias  = np.round(np.mean(orig-cannon),4)
    low = np.minimum(min(orig), min(cannon))
    high = np.maximum(max(orig), max(cannon))

    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(1,2,width_ratios=[2,1], wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.legend(fontsize=14)
    pl = ax1.scatter(orig, cannon, marker='x', c=snr, 
            vmin=50, vmax=200, alpha=0.7)
    cb = plt.colorbar(pl, ax=ax1, orientation='horizontal')
    cb.set_label('SNR from LAMOST Spectra', fontsize=12)
    cb.ax.tick_params(labelsize=14)
    textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
            fontsize=14, verticalalignment='top')
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlabel("APOGEE Label", fontsize=14)
    ax1.set_ylabel("Cannon-LAMOST Label", fontsize=14)
    ax1.set_title("Cannon-LAMOST Output vs. APOGEE Values \n Label $%s$ for SNR $\geq$ 50" % name, fontsize=14)
    diff = cannon - orig
    npoints = len(diff)
    mu = mean(diff)
    sig = std(diff)
    ax2.hist(diff, range=[-3*sig,3*sig], color='k', bins=np.sqrt(npoints), 
            orientation='horizontal', alpha=0.3, histtype='stepfilled')
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_xlabel("Count", fontsize=14)
    ax2.set_ylabel("Difference", fontsize=14)
    ax2.axhline(y=0, c='k', lw=3, label='Difference=0')
    ax2.set_title("Cannon-LAMOST Output Minus \n APOGEE Labels for $%s$" %name, 
            fontsize=14)
    ax2.legend(fontsize=14)
    plt.show()
    plt.savefig('1to1_%s.png'%i)

# first-order coeffs plot
lams = pickle.load(open("wl.p", "r"))
coeffs_all = pickle.load(open("coeffs_all.p", "r"))
scatters = pickle.load(open("scatters.p", "r"))
bad = scatters < 0.0002
scatters = np.ma.array(scatters, mask=bad)
lams = np.ma.array(lams, mask=bad)
cut = 40

lams = lams[cut:]
scatters = scatters[cut:]

nlabels = 4
fig, axarr = plt.subplots(5, figsize=(8,8), sharex=True)
#ax = plt.gca()
#ax.xaxis.grid(True)
plt.subplots_adjust(hspace=0.001)
nbins = len(ax1.get_xticklabels())
for i in range(1,5):
    axarr[i].yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
plt.xlabel(r"Wavelength $\lambda (\AA)$", fontsize=14)
plt.xlim(np.ma.min(lams), np.ma.max(lams))
plt.tick_params(axis='x', labelsize=14)
axarr[0].set_title(
    "First-Order Fit Coefficients and Scatter from the Spectral Model",
    fontsize=14)
ax.locator_params(axis='x', nbins=10)

for i in range(0,4):
    ax = axarr[i]
    lbl = r'$%s$'%names[i]
    ax.set_ylabel(lbl, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.grid(True)
    y = np.ma.array(coeffs_all[:,i+1], mask=bad)
    y = y[cut:]
    ax.step(lams, y, where='mid', linewidth=0.5, c='k')
    ax.locator_params(axis='y', nbins=4)

ax = axarr[4] 
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("scatter", fontsize=14)
top = np.max(scatters[scatters < 0.8])
stretch = np.std(scatters[scatters < 0.8])
ax.set_ylim(0, top + stretch)
ax.step(lams, scatters, where='mid', c='k', linewidth=0.7)
ax.xaxis.grid(True)
ax.locator_params(axis='y', nbins=4)
fig.savefig('leading_coeffs.png')
