# A sample run of this Cannon rewrite.

import os
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset
from aspcap import get_spectra
from aspcap import continuum_normalize
from aspcap import get_training_labels

# CONSTRUCT TRAINING SET

# Retrieve all training IDs & spectra
readin = "traininglabels.txt"
IDs = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
filenames1 = [] # if I just replace the element, I lose the '.fits'
for i in range(0, len(IDs)): # incorporate file location info
    filename = '/home/annaho/AnnaCannon/Data/APOGEE_Data' + IDs[i][1:]
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

# Optional: Set desired stars
Teff = label_values[:,0]
Teff_corr = label_values[:,1]
diff_t = np.abs(Teff-Teff_corr)
diff_t_cut = 600.
logg = label_values[:,2]
logg_cut = 100.
bad = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
IDs = IDs[bad]
spectra = spectra[bad]
label_values = label_values[bad]

# Normalize the spectra
normalized_spectra, continua = continuum_normalize(spectra)

# Initialize the training set
training_set = Dataset(IDs=IDs, spectra=spectra, label_names=label_names, label_values=label_values)

# CONSTRUCT TEST SET

# In this case, the training set and test set are the same
test_set = Dataset(IDs=IDs, spectra=spectra, label_names=label_names)

# STEP 1 OF THE CANNON: FIT FOR MODEL
model = train_model(training_set)
coeffs_all, covs, scatters, chis, chisqs, pivots = model
coeffs_all.shape # (8575, 15)
cannon_labels, MCM_rotate, covs = infer_labels(nlabels, model, test_set)

# Plot the results
cannon_labels.shape # (553, 4)
filtered_cannon_labels = cannon_labels[to_discard]
Cannon_Teff, Cannon_logg, Cannon_FeH, Cannon_age = filtered_cannon_labels[:,0], filtered_cannon_labels[:,1], filtered_cannon_labels[:,2], filtered_cannon_labels[:,3]
plt.scatter(Teff, Cannon_Teff)
# etc
