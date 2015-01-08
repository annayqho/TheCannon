# A sample run of this Cannon rewrite.

import os
from read_aspcap_2 import ReadASPCAP
from cannon1_train_model import train_model
from cannon2_infer_labels import infer_labels

training_label_names = ['Teff', 'logg', 'FeH', 'age']
nlabels = len(training_label_names)

# Extract data
trial_run = ReadASPCAP()
training_set, to_discard = trial_run.set_star_set(True, training_label_names)
test_set, nothing = trial_run.set_star_set(False, training_label_names)

# Playing with the training_set object
IDs = training_set.get_IDs()
spectra = training_set.get_spectra()
spectra.shape # (523, 8575, 3)
pixels = spectra[:,:,0]
fluxes = spectra[:,:,1]
fluxerrs = spectra[:,:,2]
plot(pixels[0], fluxes[0])
training_labels = training_set.get_label_values()
training_labels.shape # (523, 4)
Teff, logg, FeH, age = training_labels[:,0], training_labels[:,1], training_labels[:,2], training_labels[:,3]
hist(age) # plot the age distribution

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
scatter(Teff, Cannon_Teff)
# etc
