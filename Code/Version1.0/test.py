# A sample run of this Cannon rewrite.

import os
import matplotlib.pyplot as plt
import numpy as np

from aspcap import get_spectra
from aspcap import continuum_normalize
from aspcap import get_training_labels

# CONSTRUCT TRAINING SET

# Retrieve all training spectra
readin = "traininglabels.txt"
filenames = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
filenames1 = [] # for some reason if I try to replace the element,
                # it gets rid of the '.fits' at the end...very annoying
for i in range(0, len(filenames)): # incorporate file location info
    filename = '/home/annaho/AnnaCannon/Data/APOGEE_Data' + filenames[i][1:]
    filenames1.append(filename)
spectra = get_spectra(filenames1)

# Retrieve all training labels
readin = "traininglabels.txt"
label_names, label_values = get_training_labels(readin)

# Optional: Set desired training labels
cols = [1,2,3,4,5,6]
colmask = np.zeros(len(label_names), dtype=bool)
colmask[cols] = 1
label_names = [label_names[i] for i in cols]
label_values = label_values[:,colmask]

# 

trial_run = ReadASPCAP()
training_set, to_discard = trial_run.set_star_set(True, training_label_names)
test_set, nothing = trial_run.set_star_set(False, training_label_names)

# Playing with the training_set object
#IDs = training_set.get_IDs()
#spectra = training_set.get_spectra()
#spectra.shape # (523, 8575, 3)
#pixels = spectra[:,:,0]
#fluxes = spectra[:,:,1]
#fluxerrs = spectra[:,:,2]
#plot(pixels[0], fluxes[0])
training_labels = training_set.get_label_values()
#training_labels.shape # (523, 4)
Teff, logg, FeH, age = training_labels[:,0], training_labels[:,1], training_labels[:,2], training_labels[:,3]
#plt.hist(age) # plot the age distribution

# Run The Cannon
model = train_model(training_set)
coeffs_all, covs, scatters, chis, chisqs, pivots = model
coeffs_all.shape # (8575, 15)
cannon_labels, MCM_rotate, covs = infer_labels(nlabels, model, test_set)

# Plot the results
cannon_labels.shape # (553, 4)
# Note that currently, there are more test stars than training stars because of the logg & Teff cuts. In order to compare the results, we need to perform the same filtering.
filtered_cannon_labels = cannon_labels[to_discard]
Cannon_Teff, Cannon_logg, Cannon_FeH, Cannon_age = filtered_cannon_labels[:,0], filtered_cannon_labels[:,1], filtered_cannon_labels[:,2], filtered_cannon_labels[:,3]
plt.scatter(Teff, Cannon_Teff)
# etc
