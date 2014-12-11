import os
from read_aspcap import getStars
#from the_cannon import train_model, estimate_labels

training_label_names = ['Teff', 'logg', 'FeH', 'age']

training_set = getStars(training_label_names)
test_set = getStars()

#training_set = get_training_set(fitsfiles, traininglabels)
#test_set = get_test_set(fitsfiles)

#coefficients = train_model(training_set)
#newlabels = estimate_labels(coefficients, test_set)
