import os
from read_aspcap import getStars
#from the_cannon import train_model, estimate_labels

training_label_names = ['Teff', 'logg', 'FeH', 'age']

training_set = getStars(isTraining=True)
test_set = getStars(isTraining=False)

#coefficients = train_model(training_set)
#newlabels = estimate_labels(coefficients, test_set)
