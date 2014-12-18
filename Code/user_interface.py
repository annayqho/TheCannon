import os
from read_aspcap import get_stars
from cannon1_train_model import train_model
from cannon2_infer_labels import infer_labels

training_label_names = ['Teff', 'logg', 'FeH', 'age']
nlabels = len(training_label_names)

training_set, to_discard = get_stars(True, training_label_names)
test_set, nothing = get_stars(False, training_label_names)

model = train_model(training_set)
cannon_labels, MCM_rotate, covs = infer_labels(nlabels, model, test_set)

training_labels = training_set.getLabelValues()
plot(training_labels, cannon_labels)
