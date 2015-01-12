from dataset import Dataset

fts_trainingset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = [])

vesta_trainingset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = [])

cluster_trainingset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = [])

trainingset = mergesets(fts_trainingset, vesta_trainingset, cluster_trainingset)

testset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = None)

from spectral_model import SpectralModel

model = SpectralModel(label_names, order) # order = quadratic, etc
model.train(trainingset)

# If you want to stop here and just save the model...

import pickle

model.write(filename)

coeffs_all = model.coeffs # and various other attributes

from cannon_labels import CannonLabels

labels = CannonLabels(label_names)
labels.solve(model, testset)
