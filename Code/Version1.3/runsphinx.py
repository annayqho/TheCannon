"""
Compile the Sphinx documentation all at once
"""
from __future__ import (absolute_import, division, print_function)

from copy import deepcopy

from apogeedata import ApogeeDF

from cannon.dataset import (dataset_prediagnostics, dataset_postdiagnostics)
from cannon.model import CannonModel


df = ApogeeDF("example_DR10/Data",
              "example_DR10/reference_labels.csv",
              'example_DR10/contpix_lambda.txt',
              'example_DR10/pixtest4.txt')

# generate the dataset necessary to run the code
# DataFrame object will take care of book keeping necessary information
reference_set = df.dataset

# cols = [1, 3, 5]
cols = 'teff logg mh'.split()
reference_set.choose_labels(cols)

# discard object with large corrections
mask = '(abs(teff - teff_corr) < 600) & (logg < 100)'
reference_set.choose_objects(mask)

# make a test sample. Currently just use the training sample.
test_set = deepcopy(reference_set)

# Plot SNR distributions and triangle plot of reference labels
dataset_prediagnostics(reference_set, test_set)

# learn the model from the reference_set
model = CannonModel(reference_set)
model.fit() # model.train would work equivalently.

# check the model
model.diagnostics(df.contpix_file)

# infer labels with the new model for the test_set
# test_set, covs = model.infer_labels(test_set)
test_set, covs = model.predict(test_set)

# Make plots
dataset_postdiagnostics(reference_set, test_set)

cannon_set = model.draw_spectra(test_set)
model.spectral_diagnostics(test_set)
