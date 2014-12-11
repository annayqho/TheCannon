# This script uses the coefficients obtained from the Cannon training data, plus the spectra of Marie's stars, and calculates ages in Gyr as well as log(age)

# Get the coefficients from the Cannon training data
# Gyr coefficients: /home/annaho/AnnaCannon/Code/Original_Code_newLitAges/coeffs_2nd_order.pickle
# log coefficients: /home/annaho/AnnaCannon/Code/Original_Code_logAges/coeffs_2nd_order.pickle

import pickle
import os
import fitspectra_ages as f

# Get the stellar spectra using get_normalized_test_data()

f.get_normalized_training_data()

# Now, we have normed_data.pickle as output

# In fitspectra.py, the routine infer_labels_nonlinear determines the labels for a new spectrum 

fn_pickle = '/home/annaho/AnnaCannon/Code/Original_Code_newLitAges/coeffs_2nd_order.pickle'
#fn_pickle = '/home/annaho/AnnaCannon/Code/Original_Code_logAges/coeffs_2nd_order.pickle'

#f.infer_labels_nonlinear(fn_pickle, 'normed_data.pickle', 'self_2nd_order_tags.pickle', -10.950, 10.99)




