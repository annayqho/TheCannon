from __future__ import (absolute_import, division, print_function)
from lamost import LamostDataset
from cannon.model import CannonModel
from cannon.spectral_model import draw_spectra, diagnostics, triangle_pixels, overlay_spectra, residuals
import numpy as np
import pickle

###### WORKFLOW


# RUN LAMOST MUNGING CODE
dataset = LamostDataset("example_LAMOST/Training_Data",
                        "example_LAMOST/Training_Data",
                        "example_DR12/reference_labels.csv")

# Choose labels
cols = ['teff', 'logg', 'feh', 'alpha']
dataset.choose_labels(cols)

# set the headers for plotting
dataset.set_label_names_tex(['T_{eff}', '\log g', '[M/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()


# RUN CONTINUUM IDENTIFICATION CODE

# Pseudo-continuum normalization for the training spectra
norm_tr_fluxes, norm_tr_ivars = dataset.continuum_normalize_q(
        dataset.tr_fluxes, dataset.tr_ivars, q=0.90, delta_lambda=400)
# or, if it's already done
norm_tr_fluxes, norm_tr_ivars = pickle.load(open("pseudo_normed_spec.p", "r"))
# pickle.dump((norm_tr_fluxes, norm_tr_ivars), open("pseudo_normed_spec.p", "w"))

# From the cont norm training spectra, identify continuum pixels
# Identify the best 5% of continuum pixels
contmask = dataset.make_contmask(norm_tr_fluxes, norm_tr_ivars, frac=0.05)

# Identify the best 5% of continuum pixels in each of the following
# pixel regions 
dataset.ranges = [[0,50], [50,100], [100,400], [400,600], [600,1722], [1863, 1950], [1950, 2500], [2500,3000], [3000, len(dataset.wl)]]
contmask = dataset.make_contmask(norm_tr_fluxes, norm_tr_ivars, frac=0.05)
# or, if it's already done
contmask = pickle.load(open("contmask.p", "r"))

# Check it out...
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

dataset.set_continuum(contmask)

# RUN CONTINUUM NORMALIZATION CODE

dataset.ranges = [[0,1723], [1863,len(dataset.wl)]] # split into two wings
tr_cont, test_cont = dataset.fit_continuum(deg=3, ffunc="sinusoid")
# or, if it's already done
tr_cont, test_cont = pickle.load(open("cont.p", "r"))

# Check it out...
jj = 50
bad = np.var(norm_tr_ivars, axis=0) == 0
flux = np.ma.array(dataset.tr_fluxes[jj,:], mask=bad)
plot(dataset.wl, flux, alpha=0.7)
scatter(dataset.wl[contmask], flux[contmask], c='r')
cont = np.ma.array(tr_cont[jj,:], mask=bad)
plot(dataset.wl, cont)

norm_tr_fluxes, norm_tr_ivars, norm_test_fluxes, norm_test_ivars = \
        dataset.continuum_normalize_f(cont=(tr_cont, test_cont))

# Check it out...
jj = 50
bad = norm_tr_ivars[jj,:] == SMALL**2
flux = np.ma.array(norm_tr_fluxes[jj,:], mask=bad)
plot(dataset.wl, flux, alpha=0.7)

# If you approve...

dataset.tr_fluxes = norm_tr_fluxes
dataset.tr_ivars = norm_tr_ivars
dataset.test_fluxes = norm_test_fluxes
dataset.test_ivars = norm_test_ivars

# learn the model from the reference_set
model = CannonModel(dataset, 2) # 2 = quadratic model
model.fit() # model.train would work equivalently.

# check the model
model.diagnostics()

# infer labels with the new model for the test_set
dataset, label_errs = model.infer_labels(dataset)
#dataset, covs = model.predict(dataset)

# Make plots
dataset.dataset_postdiagnostics(dataset)

cannon_set = draw_spectra(model.model, dataset)
diagnostics(cannon_set, dataset, model.model)
