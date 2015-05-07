from __future__ import (absolute_import, division, print_function)
from lamost import load_spectra, load_labels
from cannon.dataset import Dataset
from cannon.model import CannonModel
from cannon.spectral_model import draw_spectra, diagnostics, triangle_pixels, overlay_spectra, residuals
import numpy as np
import pickle
import random
import csv
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# STEP 0: CHOOSE TRAINING SET

allfiles = np.loadtxt("example_LAMOST/lamost_sorted_by_ra.txt", dtype=str)
dir_dat = "example_LAMOST/Data_All"
tr_files = allfiles
wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_files)
SNR_raw = tr_flux * np.sqrt(tr_ivar)
SNR = np.median(SNR_raw[SNR_raw>0], axis=0)

# STEP 1: PREPARE DATA 
if glob.glob('lamost_data.p'):
    wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar = pickle.load(
            open('lamost_data.p', 'r'))

else:
    tr_files = np.genfromtxt("example_LAMOST/Training_Data.txt", dtype=str)
    test_files = np.loadtxt("example_LAMOST/Test_Data.txt", dtype=str)
    dir_lab = "example_DR12/reference_labels.csv"
    dir_dat = "example_LAMOST/Data_All"

    wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_files)
    wl, test_flux, test_ivar = load_spectra(dir_dat, test_files)
    tr_label = load_labels(dir_lab)
    pickle.dump((wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar), 
            open('lamost_data.p', 'w'))

dataset = Dataset(wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar)

# set the headers for plotting
dataset.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()


# STEP 2: CONTINUUM IDENTIFICATION

# Pseudo-continuum normalization for the training spectra
if glob.glob('pseudo_normed_spec.p', 'r'):
    norm_tr_fluxes, norm_tr_ivars = pickle.load(open("pseudo_normed_spec.p", "r"))

else:
    norm_tr_fluxes, norm_tr_ivars = dataset.continuum_normalize_q(
            dataset.tr_flux, dataset.tr_ivar, q=0.90, delta_lambda=400)
    pickle.dump((norm_tr_fluxes, norm_tr_ivars), 
            open("pseudo_normed_spec.p", "w"))

# From the cont norm training spectra, identify continuum pixels
if glob.glob('contmask.p', 'r'):
    contmask = pickle.load(open("contmask.p", "r"))
else:
    # Identify the best 5% of continuum pixels
    contmask = dataset.make_contmask(norm_tr_fluxes, norm_tr_ivars, frac=0.05)

    # Identify the best 5% of continuum pixels in each of the following
    # pixel regions 
    dataset.ranges = [[0,50], [50,100], [100,400], [400,600], [600,1722], [1863, 1950], [1950, 2500], [2500,3000], [3000, len(dataset.wl)]]
    contmask = dataset.make_contmask(norm_tr_fluxes, norm_tr_ivars, frac=0.05)
    # since I changed the array size...

dataset.set_continuum(contmask)


# RUN CONTINUUM NORMALIZATION CODE

if glob.glob('cont.p', 'r'):
    tr_cont, test_cont = pickle.load(open("cont.p", "r"))
else:
    dataset.ranges = [[0,1723], [1863,len(dataset.wl)]] # split into two wings
    tr_cont, test_cont = dataset.fit_continuum(deg=3, ffunc="sinusoid")
    pickle.dump((tr_cont, test_cont), open("cont.p", "w"))


# Check out the median flux overlaid with cont pix 
f_bar = np.zeros(len(dataset.wl))
sigma_f = np.zeros(len(dataset.wl))
for wl in range(0,len(dataset.wl)):
    flux = norm_tr_fluxes[:,wl]
    ivar = norm_tr_ivars[:,wl]
    f_bar[wl] = np.median(flux[ivar>0])
    sigma_f[wl] = np.sqrt(np.var(flux[ivar>0]))
bad = np.var(norm_tr_ivars, axis=0) == 0
f_bar = np.ma.array(f_bar, mask=bad)
sigma_f = np.ma.array(sigma_f, mask=bad)
plot(dataset.wl, f_bar, alpha=0.7)
fill_between(dataset.wl, (f_bar+sigma_f), (f_bar-sigma_f), alpha=0.2)
scatter(dataset.wl[contmask], f_bar[contmask], c='r')
xlim(3800,9100)
ylim(0,1.1)
xlabel("Wavelength (A)")
ylabel("Median Flux")
title("Median Flux Across Training Spectra Overlaid with Cont Pix")
savefig("medflux_contpix.png")

# Residuals between raw flux and fitted continuum
res = (dataset.tr_flux-tr_cont)*np.sqrt(dataset.tr_ivar)
# sort by temperature...
sorted_res = res[np.argsort(dataset.tr_label[:,2])]
im = plt.imshow(res, cmap=plt.cm.bwr_r, interpolation='nearest', 
        vmin = -80, vmax=10, aspect = 'auto', origin = 'lower',
        extent=[0, len(dataset.wl), 0, nstars])
xlabel("Pixel")
ylabel("Training Object")
title("Residuals in Continuum Fit to Raw Spectra")
colorbar()


# Plot the cont fits for 100 random training stars
pickstars = np.zeros(100)
nstars = dataset.tr_flux.shape[0]
for jj in range(100):
    pickstars[jj] = random.randrange(0, nstars-1)

for jj in pickstars:
    #bad = np.var(norm_tr_ivars, axis=0) == 0
    bad = norm_tr_ivars[jj,:] == 0 
    flux = np.ma.array(dataset.tr_flux[jj,:], mask=bad)
    plot(dataset.wl, flux, alpha=0.7)
    scatter(dataset.wl[contmask], flux[contmask], c='r')
    cont = np.ma.array(tr_cont[jj,:], mask=bad)
    plot(dataset.wl, cont)
    xlim(3800, 9100)
    title("Sample Continuum Fit")
    xlabel("Wavelength (A)")
    ylabel("Raw Flux")
    savefig('contfit_%s.png' %jj)
    plt.close()

f,ax = plt.subplots(2, sharex=True)
meanres = np.mean(res, axis=0)
mu = mean(meanres)
sig = np.sqrt(var(meanres))
c= ax[1].imshow(sorted_res, cmap = plt.cm.bwr_r, interpolation='nearest',
     vmin = mu-3*sig, vmax= mu+sig, aspect='auto')
colorbar(c, orientation='horizontal')
ax[0].plot(flux, alpha=0.7)
ax[0].plot(cont)
xlim(0,3626)

ax[0].hist(meanres, bins = 60, range=(mu-3*sig, mu+sig))
ax[0].set_ylabel("Raw Flux")
title("Sample Raw Spec with All Continuum Fit Residuals")
c = ax[1].imshow(res, cmap = plt.cm.bwr_r, interpolation='nearest',
        vmin = -80, vmax=10, aspect='auto')
colorbar(c, orientation='horizontal')
ax[1].set_ylabel("Training Objects")


norm_tr_fluxes, norm_tr_ivars, norm_test_fluxes, norm_test_ivars = \
        dataset.continuum_normalize_f(cont=(tr_cont, test_cont))


# If you approve...

dataset.tr_flux = norm_tr_fluxes
dataset.tr_ivar = norm_tr_ivars
dataset.test_flux = norm_test_fluxes
dataset.test_ivar = norm_test_ivars

# learn the model from the reference_set
model = CannonModel(dataset, 3) # 2 = quadratic model
model.fit() # model.train would work equivalently.

# or...
coeffs_all = pickle.load(open("coeffs_all.p", "r"))

# check the model
model.diagnostics()

# infer labels with the new model for the test_set
if glob.glob('test_labels.p'):
    test_label = pickle.load(open('test_labels.p', 'r'))
    dataset.test_label = test_label
else:
    dataset, label_errs = model.infer_labels(dataset)

# Make plots
dataset.dataset_postdiagnostics(dataset)

cannon_set = draw_spectra(model.model, dataset)
diagnostics(cannon_set, dataset, model.model)
