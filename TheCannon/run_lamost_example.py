from __future__ import (absolute_import, division, print_function)
from lamost import LamostDataset
from cannon.model import CannonModel
from cannon.spectral_model import draw_spectra, diagnostics, triangle_pixels, overlay_spectra, residuals
import numpy as np

###### WORKFLOW

# RUN LAMOST MUNGING CODE
dataset = LamostDataset("example_LAMOST/Training_Data",
                        "example_LAMOST/Training_Data",
                        "example_DR12/reference_labels.csv")

# Choose labels
cols = ['teff', 'logg', 'feh']
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

dataset.set_continuum(pseudo_cont_dataset.contmask)
dataset.ranges = [[0,1883],[2094,3899]]
tr_cont, test_cont = dataset.fit_continuum(deg=3)

# RUN CONTINUUM NORMALIZATION CODE
dataset.ranges = None
dataset.continuum_normalize(cont=(tr_cont, test_cont))

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
