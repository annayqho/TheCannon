import numpy as np
import pickle
import random
import glob
from matplotlib import rc
from cannon.model import CannonModel
from lamost import load_spectra, load_labels
from TheCannon import dataset

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# STEP 1: DATA MUNGING
allfiles = glob.glob("example_LAMOST/Data_All/*fits")
allfiles = np.char.lstrip(allfiles, 'example_LAMOST/Data_All/')
tr_ID = np.loadtxt("tr_files.txt", dtype=str)
test_ID = np.setdiff1d(allfiles, tr_ID)

dir_dat = "example_LAMOST/Data_All"
tr_IDs, wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_ID)
label_file = "reference_labels.csv"
tr_label = load_labels(label_file, tr_ID)
test_IDs, wl, test_flux, test_ivar = load_spectra(dir_dat, test_ID)

good = np.logical_and(tr_label[:,0] > 0, tr_label[:,2]>-5)
tr_IDs = tr_IDs[good]
tr_flux = tr_flux[good]
tr_ivar = tr_ivar[good]
tr_label = tr_label[good]

dataset = dataset.Dataset(
        wl, tr_IDs, tr_flux, tr_ivar, tr_label, test_IDs, test_flux, test_ivar)

# set the headers for plotting
dataset.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# STEP 2: CONTINUUM IDENTIFICATION

# Pseudo-continuum normalization for the training spectra
if glob.glob('pseudo_normed_spec.p'):
    (pseudo_flux, pseudo_ivar) = pickle.load(open("pseudo_normed_spec.p", "r"))

else:
    pseudo_flux, pseudo_ivar = dataset.continuum_normalize_training_q(
            q=0.90, delta_lambda=400)
    pickle.dump((pseudo_flux, pseudo_ivar), open("pseudo_normed_spec.p", "w"))

# From the cont norm training spectra, identify continuum pixels
if glob.glob('contmask.p', 'r'):
    contmask = pickle.load(open("contmask.p", "r"))
else:
    # Identify the best 5% of continuum pixels
    # contmask = dataset.make_contmask(norm_tr_fluxes, norm_tr_ivars, frac=0.05)

    # Identify the best 5% of continuum pixels in each of the following
    # pixel regions 
    dataset.ranges = [[0,50], [50,100], [100,400], [400,600], [600,1722], [1863, 1950], [1950, 2500], [2500,3000], [3000, len(dataset.wl)]]
    contmask = dataset.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.05)
    pickle.dump(contmask, open("contmask.p", "w"))

dataset.set_continuum(contmask)


# RUN CONTINUUM NORMALIZATION CODE
dataset.ranges = [[0,1723], [1863,len(dataset.wl)]] # split into two wings

if glob.glob('cont.p', 'r'):
    cont = pickle.load(open("cont.p", "r"))
else:
    cont = dataset.fit_continuum(deg=3, ffunc="sinusoid")
    pickle.dump((cont), open("cont.p", "w"))

norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
        dataset.continuum_normalize(cont)

dataset.tr_flux = norm_tr_flux
dataset.tr_ivar = norm_tr_ivar
dataset.test_flux = norm_test_flux
dataset.test_ivar = norm_test_ivar

# learn the model from the reference_set
from TheCannon import model
model = model.CannonModel(dataset, 2) # 2 = quadratic model
model.fit() # model.train would work equivalently.
pickle.dump(coeffs_all, open("coeffs_all.p", "w"))

# or...
coeffs_all = pickle.load(open("coeffs_all.p", "r"))

# check the model
model.diagnostics()

# infer labels with the new model for the test_set
if glob.glob('test_labels.p'):
    test_label = pickle.load(open('test_labels.p', 'r'))
    dataset.test_label = test_label
else:
    label_errs = model.infer_labels(dataset)

# Make plots
dataset.dataset_postdiagnostics(dataset)

cannon_set = draw_spectra(model.model, dataset)
diagnostics(cannon_set, dataset, model.model)
