import numpy as np
import pickle
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

data = dataset.Dataset(
        wl, tr_IDs, tr_flux, tr_ivar, tr_label, test_IDs, test_flux, test_ivar)

# set the headers for plotting
data.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

# Plot SNR distributions and triangle plot of reference labels
data.diagnostics_SNR()
data.diagnostics_ref_labels()

# STEP 2: CONTINUUM NORMALIZATION 
data.continuum_normalize_gaussian_smoothing(L=50)

# learn the model from the reference_set
model = model.CannonModel(2) # 2 = quadratic model
model.fit(dataset) # model.train would work equivalently.
pickle.dump(coeffs_all, open("coeffs_all.p", "w"))

# or...
coeffs_all = pickle.load(open("coeffs_all.p", "r"))

# check the model
model.diagnostics(dataset)

# infer labels with the new model for the test_set
if glob.glob('test_labels.p'):
    test_label = pickle.load(open('test_labels.p', 'r'))
    data.test_label = test_label
else:
    model.infer_labels(data)

# Make plots
data.diagnostics_1to1()
data.survey_label_triangle()
