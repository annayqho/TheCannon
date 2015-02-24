from __future__ import (absolute_import, division, print_function)
from apogee import ApogeeDataset
from cannon.model import CannonModel

###### WORKFLOW

# RUN APOGEE MUNGING CODE
dataset = ApogeeDataset("example_DR10/Data",
                        "example_DR10/Data",
                        "example_DR10/reference_labels.csv")

# Choose labels
cols = ['teff', 'logg', 'mh']
dataset.choose_labels(cols)

# set the headers for plotting
dataset.set_label_names_tex(['T_{eff}', '\log g', '[M/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# RUN CONTINUUM IDENTIFICATION CODE
dataset.find_continuum()

# RUN CONTINUUM NORMALIZATION CODE
dataset.continuum_normalize()

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
