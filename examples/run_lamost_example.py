import numpy as np
import pickle
import random
import glob
from matplotlib import rc
from lamost import load_spectra, load_labels
from TheCannon import dataset
from TheCannon import model

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# STEP 1: DATA MUNGING
allfiles = glob.glob("example_LAMOST/Data_All/*fits")
allfiles = np.char.lstrip(allfiles, 'example_LAMOST/Data_All/')
# or...tr_ID = np.loadtxt("badstars.txt", dtype=str)
tr_ID = np.loadtxt("tr_files.txt", dtype=str)
test_ID = np.setdiff1d(allfiles, tr_ID)

dir_dat = "example_LAMOST/Data_All"
tr_IDs, wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_ID)

label_file = "apogee_dr12_labels.csv"
all_labels = load_labels(label_file, tr_ID)
teff = all_labels[:,0]
logg = all_labels[:,1]
mh = all_labels[:,2]
alpha = all_labels[:,3]
tr_label = np.vstack((teff, logg, mh, alpha)).T

test_IDs, wl, test_flux, test_ivar = load_spectra(dir_dat, test_ID)

dataset = dataset.Dataset(
        wl, tr_IDs, tr_flux, tr_ivar, tr_label, test_IDs, test_flux, test_ivar)

# set the headers for plotting
dataset.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# STEP 2: CONTINUUM IDENTIFICATION



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
