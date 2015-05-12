from __future__ import (absolute_import, division, print_function)
from apogee import load_spectra, load_labels
from cannon.model import CannonModel
from cannon.spectral_model import draw_spectra, diagnostics, triangle_pixels, overlay_spectra, residuals
import numpy as np

###### WORKFLOW

# PREPARE DATA

wl, tr_flux, tr_ivar = load_spectra("example_DR10/Data")
test_flux = tr_flux
test_ivar = tr_ivar
tr_label = load_labels("example_DR10/reference_labels.csv")
dataset = Dataset(wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar)
dataset.ranges = [[371,3192], [3697,5997], [6461,8255]]

# optional: set headers for plotting
dataset.set_label_names_tex(['T_{eff}', '\log g', '[M/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# RUN CONTINUUM IDENTIFICATION CODE
pixlist = np.array(
      np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1, dtype=int))
npix = len(dataset.wl)
contmask = np.zeros(npix, dtype=bool)
contmask[pixlist] = True
# get rid of the contpix that are in the gap
gapmask = dataset.find_gaps(dataset.tr_fluxes)
contmask[gapmask] = False
dataset.set_continuum(contmask)

# RUN CONTINUUM NORMALIZATION CODE
# with running quantile
# dataset.continuum_normalize(q=0.90)

# with Chebyshev fit
dataset.continuum_normalize()

# learn the model from the reference_set
model = CannonModel(dataset, 2) # 2 = quadratic model
model.fit() # model.train would work equivalently.

# check the model
#model.diagnostics()

# infer labels with the new model for the test_set
dataset, label_errs = model.infer_labels(dataset)
#dataset, covs = model.predict(dataset)

# Make plots
dataset.dataset_postdiagnostics(dataset)

# cannon_set = model.draw_spectra(dataset)
# model.spectral_diagnostics(dataset)
