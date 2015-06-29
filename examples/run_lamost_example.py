import numpy as np
import pickle
import glob
from matplotlib import rc
from lamost import load_spectra, load_labels
from TheCannon import continuum_normalization
from TheCannon import dataset
from TheCannon import model

rc('text', usetex=True)
rc('font', family='serif')

# STEP 1: DATA MUNGING
tr_ID = np.loadtxt("tr_files.txt", dtype=str)
test_ID = np.loadtxt("lamost_dr2/20121125_filenames.txt", dtype=str)

if glob.glob("tr_data_raw.npz"):
    print("training data already loaded")
    with np.load("tr_data_raw.npz") as data:
        tr_IDs = data['arr_0']
        wl = data['arr_1']
        tr_flux = data['arr_2']
        tr_ivar = data['arr_3']
else:
    dir_dat = "example_LAMOST/Data_All"
    tr_IDs, wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_ID)
    np.savez("./tr_data_raw", tr_IDs, wl, tr_flux, tr_ivar)

label_file = "apogee_dr12_labels.csv"
all_labels = load_labels(label_file, tr_IDs)
teff = all_labels[:,0]
logg = all_labels[:,1]
mh = all_labels[:,2]
alpha = all_labels[:,3]
test_label = np.vstack((teff, logg, mh, alpha)).T
np.savez("./tr_label.npz", tr_label)
with np.load("tr_label.npz") as a:
    tr_label = a['arr_0']

if glob.glob("test_data_raw.npz"):
else:
    dir_dat = "lamost_dr2/DR2_release"
    test_IDs, wl, test_flux, test_ivar = load_spectra(dir_dat, test_ID)
    np.savez("./test_data_raw", test_IDs, wl, test_flux, test_ivar)

with np.load("tr_norm.npz") as a:
    tr_flux = a['arr_0']
    tr_ivar = a['arr_1']
with np.load("tr_IDs.npz") as a:
    tr_IDs = a['arr_0']
with np.load("wl.npz") as a:
    wl = a['arr_0']

data = dataset.Dataset(
        wl, tr_IDs, tr_flux, tr_ivar, tr_label, test_IDs, test_flux, test_ivar)

# set the headers for plotting
data.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

# Plot SNR distributions and triangle plot of reference labels
data.diagnostics_SNR()
data.diagnostics_ref_labels()

# STEP 2: CONTINUUM NORMALIZATION 
if glob.glob("tr_data_norm"):
    print("data already normalized")
    with np.load("tr_norm.npz") as a:
        tr_flux = a['arr_0']
        tr_ivar = a['arr_1']
    with np.load("test_norm.npz") as a:
        test_flux = a['arr_0']
        test_ivar = a['arr_1']
    data.tr_flux = tr_flux
    data.tr_ivar = tr_ivar
    data.test_flux = test_flux
    data.test_ivar = test_ivar

else:
    data.continuum_normalize_gaussian_smoothing(L=50)
    np.savez("./tr_norm", data.tr_flux, data.tr_ivar)
    np.savez("./test_norm", data.test_flux, data.test_ivar)

# add PS1 colors
# ids = np.loadtxt("example_PS1/ps_colors.txt", usecols=(0,), dtype='str', delimiter=',')
# colors_all = np.loadtxt("example_PS1/ps_colors.txt", usecols=(1,2,3,4), dtype='float', delimiter=',')
# apogee_ids = np.loadtxt("apogee_dr12_labels.csv", dtype='str', usecols=(1,), delimiter=',')
# apogee_ids_short = np.array([apogee_id[19:37] for apogee_id in apogee_ids])
# inds = np.array([np.where(ids==apogee_id)[0][0] for apogee_id in apogee_ids_short])
# colors = colors_all[inds]

# add another column to the tr_flux, tr_ivar, test_flux, test_ivar

# learn the model from the reference_set
model = model.CannonModel(2) # 2 = quadratic model
model.fit(data) # model.train would work equivalently.
np.savez("./model_coeffs", model.coeffs)
np.savez("./model_scatter", model.scatters)
np.savez("./model_chisqs", model.chisqs)
np.savez("./model_pivots", model.pivots)
model.coeffs = np.load("./model_coeffs.npz")['arr_0']
model.scatters = np.load("./model_scatter.npz")['arr_0']
model.chisqs = np.load("./model_chisqs.npz")['arr_0']
model.pivots = np.load("./model_pivots.npz")['arr_0']

# check the model
# model.diagnostics(dataset)

# infer labels with the new model for the test_set
#else:
model.infer_labels(data)
np.savez("./test_labels.npz", data.test_label_vals)
with np.load("test_label.npz") as a:
    test_label = a['arr_0']

# Make plots
data.diagnostics_1to1()
data.survey_label_triangle()

# Find chi sq of fit

lvec = _get_lvec(list(data.test_label_vals[jj,:]-model.pivots))
chi = data.tr_flux[jj,:] - (np.dot(coeffs, lvec) + model.coeffs[:,0])
chisq_star = sum(chi**2)
dof = npix - nparam
