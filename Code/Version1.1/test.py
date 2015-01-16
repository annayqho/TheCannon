# A sample run of this Cannon rewrite.

import os
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset
from dataset import training_set_diagnostics
from dataset import test_set_diagnostics
from read_apogee import get_spectra
from read_apogee import continuum_normalize
from read_apogee import get_training_labels
from cannon1_train_model import train_model
from cannon1_train_model import model_diagnostics
from cannon2_infer_labels import infer_labels

# CONSTRUCT TRAINING SET
readin = "traininglabels.txt"

# Retrieve all training IDs & spectra
IDs = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
filenames1 = [] # if I just replace the element, I lose the '.fits'
for i in range(0, len(IDs)): # incorporate file location info
    filename = '/home/annaho/AnnaCannon/Data/APOGEE_Data' + IDs[i][1:]
    filenames1.append(filename)
spectra, SNRs = get_spectra(filenames1)

# Retrieve all training labels
all_label_names, all_label_values = get_training_labels(readin)

# Optional: Set desired training labels
cols = [1, 3, 5]
colmask = np.zeros(len(all_label_names), dtype=bool)
colmask[cols] = 1
label_names = [all_label_names[i] for i in cols]
label_values = all_label_values[:,colmask]

dataset.choose_spectra(mask)

# Optional: Set desired stars
Teff = label_values[:,0]
Teff_corr = all_label_values[:,2]
diff_t = np.abs(Teff-Teff_corr)
diff_t_cut = 600.
logg = label_values[:,1]
logg_cut = 100.
bad = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
IDs = IDs[bad]
spectra = spectra[bad]
label_values = label_values[bad]

# Normalize the spectra
normalized_spectra, continua = continuum_normalize(spectra)

# Initialize the training set 
training_set = Dataset(IDs=IDs, SNRs=SNRs, spectra=normalized_spectra, 
        label_names=label_names, label_values=label_values)
training_set_diagnostics(training_set)

# CONSTRUCT TEST SET

# In this case, the training set and test set are the same
test_set = Dataset(IDs=IDs, SNRs = SNRs, spectra=normalized_spectra, 
        label_names=label_names)

# STEP 1 OF THE CANNON: FIT FOR MODEL
print "training model"
model, label_vector = train_model(training_set)
print "done training model"
model_diagnostics(training_set, model, label_vector)

# coeffs_all, covs, scatters, chis, chisqs, pivots = model

# STEP 2 OF THE CANNON: INFER LABELS
nlabels = len(label_names)
cannon_labels, MCM_rotate, covs = infer_labels(nlabels, model, test_set)
test_set.set_label_values(cannon_labels)
test_set_diagnostics(training_set, test_set)

# Finally, create our artificial "Cannon Set"
nstars = len(test_set.IDs)
cannon_spectra = np.zeros(test_set.spectra.shape)
cannon_spectra[:,:,0] = test_set.spectra[:,:,0]
for i in range(nstars):
    x = label_vector[:,i,:]
    spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
    cannon_spectra[i,:,1]=spec_fit
cannon_set = Dataset(IDs=IDs, SNRs=SNRs, spectra=cannon_spectra, 
        label_names = label_names, label_values = cannon_labels)