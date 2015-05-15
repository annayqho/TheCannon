import numpy as np
import glob
from TheCannon import apogee
from TheCannon import dataset
from TheCannon import model 
import pickle

# (1) PREPARE DATA

tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("example_DR10/Data")
# doing a 1-to-1 test for simplicity
test_ID = tr_ID
test_flux = tr_flux 
test_ivar = tr_ivar
all_labels = apogee.load_labels("example_DR10/reference_labels.csv")
# choose labels and make a new array 
teff_corr = all_labels[:,1] 
logg_corr = all_labels[:,3]
mh_corr = all_labels[:,5]
tr_label = np.vstack((teff_corr, logg_corr, mh_corr)).T
dataset = dataset.Dataset(wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)
# apogee spectra come in three segments, corresponding to the three chips
dataset.ranges = [[371,3192], [3697,5997], [6461,8255]]

# set LaTeX label names for making diagnostic plots
dataset.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# (2) IDENTIFY CONTINUUM PIXELS

# pseudo continuum normalize the spectrum using a running quantile
if glob.glob('pseudo_spec.p'):
    (pseudo_tr_flux, pseudo_tr_ivar) = pickle.load(open('pseudo_spec.p', 'r'))
else:
    pseudo_tr_flux, pseudo_tr_ivar = dataset.continuum_normalize_training_q(
            q=0.90, delta_lambda=50)
    pickle.dump((pseudo_tr_flux, pseudo_tr_ivar), open("pseudo_spec.p", 'w'))

# in each region of the pseudo cont normed tr spectrum, 
# identify the best 7% of continuum pix
if glob.glob('contmask.p'):
    contmask = pickle.load(open('contmask.p', 'r'))
else:
    contmask = dataset.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)
    pickle.dump(contmask, open('contmask.p', 'w'))

dataset.set_continuum(contmask)

# fit a sinusoid through the continuum pixels
if glob.glob('cont.p'):
    cont = pickle.load(open('cont.p', 'r')) 
else:
    cont = dataset.fit_continuum(3, "sinusoid")
    pickle.dump(cont, open('cont.p', 'w'))

# (3) CONTINUUM NORMALIZE
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
        dataset.continuum_normalize_f(cont)

# replace with normalized values
dataset.tr_flux = norm_tr_flux
dataset.tr_ivar = norm_tr_ivar
dataset.test_flux = norm_test_flux
dataset.test_ivar = norm_test_ivar

# (4) TRAINING STEP

# learn the model from the reference_set
model = model.CannonModel(dataset, 2) # 2 = quadratic model
model.fit() # model.train would work equivalently.
model.diagnostics()

# (5) TEST STEP

# infer labels with the new model for the test_set
label_errs = model.infer_labels(dataset)
dataset.diagnostics_test_step_flagstars()
dataset.diagnostics_survey_labels()
dataset.diagnostics_1to1()
