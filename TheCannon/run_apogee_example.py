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
contmask = dataset.find_continuum()

# RUN CONTINUUM NORMALIZATION CODE
dataset.continuum_normalize(contmask)

# learn the model from the reference_set
model = CannonModel(dataset)
model.fit() # model.train would work equivalently.

# check the model
model.diagnostics()

# infer labels with the new model for the test_set
test_set, covs = model.infer_labels(test_set)
#test_set, covs = model.predict(test_set)

# Make plots
#dataset_postdiagnostics(reference_set, test_set)

#cannon_set = model.draw_spectra(test_set)
#model.spectral_diagnostics(test_set)
