from __future__ import (absolute_import, division, print_function)
from apogee import load_spectra, load_labels
from cannon.model import CannonModel
from cannon.dataset import Dataset
from cannon.spectral_model import draw_spectra, diagnostics, triangle_pixels, overlay_spectra, residuals
import numpy as np

# (1) PREPARE DATA

wl, tr_flux, tr_ivar = load_spectra("example_DR10/Data")
test_flux = tr_flux
test_ivar = tr_ivar
all_labels = load_labels("example_DR10/reference_labels.csv")
teff_corr = all_labels[:,1]
logg_corr = all_labels[:,3]
mh_corr = all_labels[:,5]
tr_label = np.vstack((teff_corr, logg_corr, mh_corr)).T
dataset = Dataset(wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar)
dataset.ranges = [[371,3192], [3697,5997], [6461,8255]]

# optional: set headers for plotting
dataset.set_label_names(['T_{eff}', '\log g', '[M/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# (2) IDENTIFY CONTINUUM PIXELS

pseudo_tr_flux, pseudo_tr_ivar = dataset.continuum_normalize_training_q(
        q=0.90, delta_lambda=50)

# in each region of the pseudo cont normed tr spectrum, 
# identify the best 7% of continuum pix
contmask = dataset.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)
dataset.set_continuum(contmask)
cont = dataset.fit_continuum(3, "sinusoid")

# (3) RUN CONTINUUM NORMALIZATION CODE
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
        dataset.continuum_normalize_f(cont)

dataset.tr_flux = norm_tr_flux
dataset.tr_ivar = norm_tr_ivar
dataset.test_flux = norm_test_flux
dataset.test_ivar = norm_test_ivar

# (4) TRAINING STEP

# learn the model from the reference_set
model = CannonModel(dataset, 2) # 2 = quadratic model
model.fit() # model.train would work equivalently.

# check the model
model.diagnostics()

# (5) TEST STEP

# infer labels with the new model for the test_set
dataset, label_errs = model.infer_labels(dataset)

# Make plots
dataset.dataset_postdiagnostics(dataset)

cannon_set = model.draw_spectra(dataset)
# model.spectral_diagnostics(dataset)
