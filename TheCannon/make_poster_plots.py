from __future__ import (absolute_import, division, print_function)
from lamost import LamostDataset
from cannon.model import CannonModel
from cannon.spectral_model import draw_spectra, diagnostics, triangle_pixels, overlay_spectra, residuals
import numpy as np
import pickle
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tr_files = np.genfromtxt("example_LAMOST/Training_Data.txt", dtype=str)
test_files = np.loadtxt("example_LAMOST/Test_Data.txt", dtype=str)
dataset = LamostDataset("example_LAMOST/Data_All",
                        tr_files, test_files,
                        "example_DR12/reference_labels.csv")

# Make a nice plot of a typical high-S/N spectrum
# with a continuum fit

# SNR of jj=0 is 134.13631569901949
# SNR of jj=150 is 172.30452688124373
# SNR of jj=10 is 280.99611610353145

tr_cont, test_cont = pickle.load(open("cont.p", "r"))

jj = 0 
bad1 = dataset.tr_fluxes[jj,:] == 0.
bad2 = np.var(dataset.tr_ivars, axis=0) == 0.
bad = np.logical_or(bad1, bad2)
x = dataset.wl
fluxes = np.ma.array(dataset.tr_fluxes[jj,:], mask=bad)
cont = np.ma.array(tr_cont[jj,:], mask=bad)
x = np.ma.array(dataset.wl, mask=bad)
plot(x, fluxes, c='k', alpha=0.7, label="Raw Spectrum")
plot(x, cont, c='r', alpha=0.7, label="Cannon Continuum Fit")
title(r"Typical High-S/N LAMOST Spectrum", fontsize=27)
xlim(3500, 9500)
tick_params(axis='x', labelsize=27)
tick_params(axis='y', labelsize=27)
xlabel("Wavelength ($\AA$)", fontsize=27)
ylabel("Flux", fontsize=27)
legend(loc='bottom right')
tight_layout()
savefig("poster_typical_spec_snr134_withcont.png")

import pickle
coeffs_all = pickle.load(open("coeffs_all.p", "r"))
c = ['k', 'b', 'g', 'r']
nlabels = 4
stds = np.array([np.std(coeffs_all[:, i + 1]) for i in range(nlabels)])
pivot_std = max(stds)
ratios = np.round(pivot_std / stds, -1)  # round to the nearest 10
ratios[ratios == 0] = 1
ratios[3] = 5 # alpha/Fe
ratios[1] = 5 # logg
ratios[2] = 5 # Fe/H
dataset.set_label_names_tex(['T_{eff}', '\log g', '[M/H]', 'alpha'])
label_names = dataset.get_plotting_labels()
label_names[3] = '\\alpha/Fe'
scatters = pickle.load(open("scatters.p", "r"))
bad = scatters < 0.0002
scatters = np.ma.array(scatters, mask=bad)
lams = np.ma.array(lams, mask=bad)

fig, axarr = plt.subplots(2, sharex=True)
plt.xlabel(r"Wavelength $\lambda (\AA)$", fontsize=27)
plt.xlim(np.ma.min(lams), np.ma.max(lams))
ax = axarr[0]
ax.set_ylabel(r"$b_\lambda, c_\lambda, d_\lambda, e_\lambda$", fontsize=27)
ax.set_title("First-Order Fit Coefficients for Labels", fontsize=27)
npixels = len(lams)
first_order = np.zeros((npixels, nlabels))
lbl = r'${0:s}_\lambda$=coeff for ${1:s}$ * ${2:d}$'
lett = ['b', 'c', 'd', 'e']
tick_params(axis='x', labelsize=20)
tick_params(axis='y', labelsize=20)
for i in range(nlabels):
    coeffs = coeffs_all[:,i+1] * ratios[i]
    coeffs = np.ma.array(coeffs, mask=bad)
    ax.plot(lams, coeffs, c=c[i], linewidth=0.5, alpha=1,
            label=lbl.format(lett[i], label_names[i], int(ratios[i])))
    box = ax.get_position()
    ax.set_position(
            [box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
    ax.legend(
            bbox_to_anchor=(0., 0, 1., .102), loc=3, ncol=4,
            mode="expand", prop={'family':'serif', 'size':15})
ax.set_ylim(-2,1.6)
ax = axarr[1]
ax.set_ylabel("scatter", fontsize=20)
tick_params(axis='y', labelsize=20)
xlim(4000,9000)
ax.set_ylim(0,0.10)
ax.step(lams, scatters, where='mid', c='k', linewidth=0.7)
fig.tight_layout(pad=2.0, h_pad=4.0)

test_labels = pickle.load(open("test_labels.p", "r"))
# choose 1000 of them
feh = test_labels[:,2]
alpha = test_labels[:,3]
feh = np.array(apogee_label_vals[:,3], dtype=float)
alpha = np.array(apogee_label_vals[:,4], dtype=float)
snr_lamost = dataset.test_SNRs
snr_apogee = np.array(apogee_label_vals[:,5], dtype=float)
vscat = np.array(apogee_label_vals[:,6], dtype=float)
flag = np.array(apogee_label_vals[:,7], dtype=bool)
good1 = np.logical_and(feh > -8000, snr_lamost > 50)
good2 = np.logical_and(vscat < 1.0, flag==0)
good3 = snr_apogee > 300 
good12 = np.logical_and(good1, good2)
good = np.logical_and(good12, good3)
feh2 = feh[good]
alpha2 = alpha[good]
snr2 = snr[good]
vscat2 = vscat[good]
flag2 = flag[good]
from matplotlib.colors import LogNorm
scatter(feh2, alpha2, c=snr2, marker='x', alpha=0.5, vmin=50, vmax=500)
a = hist2d(feh2, alpha2, bins=60, range=[[-2.5, 1.0],[-0.2,0.6]], norm=LogNorm(), vmin=1, vmax=50)
tick_params(axis='x', labelsize=20)
tick_params(axis='y', labelsize=20)
ylabel(r"$[\alpha/Fe]$", fontsize=27)
xlabel(r"$[Fe/H]$", fontsize=27)
title(r"$[Fe/H]-[\alpha/Fe]$ Distribution for \newline Test Objects, ASPCAP Values", fontsize=27)
xlim(-2,1)
ylim(-0.2,0.5)

# 1-to-1 plots
from cannon.helpers import Table
label_file = "example_DR12/apogee_test_labels.csv"
data = Table(label_file)
data.sort('id')
label_names = data.keys()
nlabels = len(label_names)
apogee_label_vals = np.array([data[k] for k in label_names]).T

# for training set
names = dataset.get_plotting_labels()
names[3] = '[\\alpha/Fe]'
names[2] = '[Fe/H]'
snr = dataset.tr_SNRs

i = 3
name = names[i]
orig = reference_labels[:,i]
cannon = test_labels[:,i]

# for HW:
bad = orig < -8000 
good = snr > 100
orig = np.ma.array(orig, mask=bad)
cannon = np.ma.array(cannon, mask=bad)
snr = np.ma.array(snr, mask=bad)
orig = orig[good]
cannon = cannon[good]
snr = snr[good]
hist(cannon-orig)
scatter = np.round(np.std(orig-cannon),3)
bias  = np.round(np.mean(orig-cannon),3)
low = np.minimum(min(orig), min(cannon))
high = np.maximum(max(orig), max(cannon))
fig, axarr = plt.subplots(2)
ax1 = axarr[0]
ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
s = ax1.scatter(orig, cannon, marker='x', c=snr, vmin=100, vmax=200)
cb = plt.colorbar(s, ax=ax1)
cb.set_label('SNR')
textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
ax1.text(0.15, 0.85, textstr, transform=ax1.transAxes, fontsize=25,
        verticalalignment='top')
ax1.set_xlabel("APOGEE Training Label")
ax1.set_ylabel("LAMOST Test Label")
ax1.set_title("1-1 Plot of Label " + r"$%s$" % name)
ax2 = axarr[1]
ax2.hist(cannon-orig, range=[-200,200])
ax2.set_xlabel("LAMOST Test Value-APOGEE Training Value")
ax2.set_ylabel("Count")
ax2.set_title("Histogram of Output Minus Ref Labels")
tight_layout()
savefig("TrainingSet_1to1_alpha.png")

# for my poster:
i = 3
cannon = np.array(test_labels[:,i])
orig = np.array(apogee_label_vals[:,i+1], dtype=float)
snr = dataset.test_SNRs
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
bias  = np.round(np.mean(orig-cannon),3)
low = np.minimum(min(orig), min(cannon))
high = np.maximum(max(orig), max(cannon))
plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
xlim(low, high)
ylim(low, high)
plt.scatter(orig, cannon, marker='x', c=snr, vmin=50, vmax=200, alpha=0.7)
cb = colorbar()
cb.set_label('SNR', fontsize=25)
cb.ax.tick_params(labelsize=25)
textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
text(0.20, 1.2, textstr, transform=ax1.transAxes, fontsize=25, verticalalignment='top')
tick_params(axis='x', labelsize=25)
tick_params(axis='y', labelsize=25)
xlabel("APOGEE Label", fontsize=25)
ylabel("Cannon-LAMOST Label", fontsize=25)
title("Cannon Output vs. APOGEE Values \n Label $%s$ for SNR $\geq$ 50" % name, fontsize=25)
tight_layout()
hist(cannon-orig, range=[-300,300], bins=20)
tick_params(axis='x', labelsize=25)
tick_params(axis='y', labelsize=25) 
xlabel("LAMOST Test Value-APOGEE Training Value", fontsize=25)
ylabel("Count", fontsize=25)
title("Histogram of Cannon-LAMOST Output \n Minus APOGEE Labels for $%s$" %name, fontsize=25)
tight_layout()

##

teff_training = np.array(dataset.tr_label_vals[:,1], dtype=float)
teff_test = test_labels[:,0]
logg_training = np.array(dataset.tr_label_vals[:,2], dtype=float)
logg_test = test_labels[:,1]
feh_training = np.array(dataset.tr_label_vals[:,3], dtype=float)
feh_test = test_labels[:,2]

plt.scatter(teff_test, logg_test, c=feh_test, marker='x', vmin=-2, vmax=1, label='Test Objects')
plt.scatter(teff_training, logg_training, c=feh_training, marker='*', edgecolor='black', linewidth='0.5', vmin=-2, vmax=1, label="Training Objects")
title(r"$T_{eff}-\log g$ Plane, Color-Coded by $[Fe/H]$", fontsize=25)
ylabel(r"$\log g$ from Training Values, dex", fontsize=25)
xlabel(r"$T_{eff}$ from Training Values, [K]", fontsize=25)
legend(loc='upper left')
cb = colorbar()
cb.ax.tick_params(labelsize=25)
tick_params(axis='x', labelsize=25)
tick_params(axis='y', labelsize=25)
cb.set_label(r"$[Fe/H]$", fontsize=25)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()



