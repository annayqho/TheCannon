from __future__ import (absolute_import, division, print_function)
from lamost import LamostDataset
from cannon.model import CannonModel
import numpy as np

###### WORKFLOW

# RUN APOGEE MUNGING CODE
dataset = LamostDataset("example_LAMOST/Testing",
                        "example_LAMOST/Testing",
                        "example_DR12/reference_labels.csv")

# Choose labels
cols = ['teff', 'logg', 'feh']
dataset.choose_labels(cols)

# set the headers for plotting
dataset.set_label_names_tex(['T_{eff}', '\log g', '[M/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# Pseudo-continuum normalization
dataset.continuum_normalize(q=0.50, delta_lambda=90)

# RUN CONTINUUM IDENTIFICATION CODE
dataset.find_continuum(f_cut=0.003, sig_cut=0.003)
# pixlist = np.array(
#        np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1, dtype=int))
#npix = len(dataset.wl)
#contmask = np.zeros(npix, dtype=bool)
#contmask[pixlist] = True
# get rid of the contpix that are in the gap
#gapmask = dataset.find_gaps(dataset.tr_fluxes)
#contmask[gapmask] = False
#dataset.set_continuum(contmask)


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

cannon_set = model.draw_spectra(dataset)
model.spectral_diagnostics(dataset)
