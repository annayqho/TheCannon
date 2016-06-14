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

with np.load("test_data_raw.npz") as data:
    test_IDs = data['arr_0']
    wl = data['arr_1']
    test_flux = data['arr_2']
    test_ivar = data['arr_3']

data = dataset.Dataset(
        wl, test_IDs[0:10], test_flux[0:10,:], test_ivar[0:10,:], [1], test_IDs, test_flux, test_ivar)

data.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

data.continuum_normalize_gaussian_smoothing(L=50)
np.savez("./test_norm", test_IDs, wl, data.test_flux, data.test_ivar)
