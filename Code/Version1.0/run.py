# Run the Cannon.

import os
from read_aspcap_2 import ReadASPCAP
from cannon1_train_model import train_model
from cannon2_infer_labels import infer_labels

training_label_names = ['Teff', 'logg', 'FeH', 'age']
print "Running with training labels:"
print training_label_names
nlabels = len(training_label_names)

# Extract data
trial_run = ReadASPCAP()
print "Running..."
print "training_set, to_discard = trial_run.set_star_set(True, training_label_names)"
training_set, to_discard = trial_run.set_star_set(True, training_label_names)
print "Running..."
print "test_set, nothing = trial_run.set_star_set(False, training_label_names)"
test_set, nothing = trial_run.set_star_set(False, training_label_names)
print "Acquiring training labels"
training_labels = training_set.get_label_values()
Teff, logg, FeH, age = training_labels[:,0], training_labels[:,1], training_labels[:,2], training_labels[:,3]

# Run The Cannon
print "Running The Cannon"
print "Step 1: model = train_model(training_set)"
model = train_model(training_set)
print "Step 2: cannon_labels, MCM_rotate, covs = infer_labels(nlabels, model, test_set)"
cannon_labels, MCM_rotate, covs = infer_labels(nlabels, model, test_set)

print "Done, printing results"

# Plot the results
filtered_cannon_labels = cannon_labels[to_discard]
Cannon_Teff, Cannon_logg, Cannon_FeH, Cannon_age = filtered_cannon_labels[:,0], filtered_cannon_labels[:,1], filtered_cannon_labels[:,2], filtered_cannon_labels[:,3]
scatter(age, Cannon_age)
# etc
