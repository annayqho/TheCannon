from __future__ import (absolute_import, division, print_function)

#from copy import deepcopy

from apogee_munging import ApogeeDataset

#from cannon.dataset import (dataset_prediagnostics, dataset_postdiagnostics)
#from cannon.model import CannonModel

# Three inputs: dir w/ training files, dir w/ test files, file w/ tr labels
# also initializes self.ranges, which tell you the ranges containing data
dataset = ApogeeDataset("example_DR10/Data",
                        "example_DR10/Data",
                        "example_DR10/reference_labels.csv")

# cols = [1, 3, 5]
# cols = 'teff logg mh'.split()
# reference_set.choose_labels(cols)

# set the headers for plotting
# reference_set.set_label_names_tex(['T_{eff}', '\log g', '[M/H]'])

# make a test sample. Currently just use the training sample.
# test_set = df.dataset

# Plot SNR distributions and triangle plot of reference labels
#dataset_prediagnostics(reference_set, test_set)

# learn the model from the reference_set
#model = CannonModel(reference_set)
#model.fit() # model.train would work equivalently.

# check the model
#model.diagnostics()

# infer labels with the new model for the test_set
# test_set, covs = model.infer_labels(test_set)
#test_set, covs = model.predict(test_set)

# Make plots
#dataset_postdiagnostics(reference_set, test_set)

#cannon_set = model.draw_spectra(test_set)
#model.spectral_diagnostics(test_set)
