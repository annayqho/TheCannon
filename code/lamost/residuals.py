""" Calculate LAMOST residuals for high-SNR objects
to search for Li-rich giants for Andy Casey """

import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
from matplotlib.ticker import MaxNLocator
from TheCannon import model
from TheCannon import dataset

DATA_DIR = "/Users/annaho/Data/Mass_And_Age"

snr = np.load("%s/ref_snr.npz" %DATA_DIR)['arr_0']
choose = snr > 100

wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
test_label = np.load("%s/xval_cannon_label_vals.npz" %DATA_DIR)['arr_0']
ds.test_label_vals = test_label
tr_flux = np.load("%s/ref_flux.npz" %DATA_DIR)['arr_0']
tr_ivar = np.load("%s/ref_ivar.npz" %DATA_DIR)['arr_0']
ds.test_flux = tr_flux
ds.test_ivar = tr_ivar

m = model.CannonModel(2)
m.coeffs = np.load("%s/coeffs.npz" %DATA_DIR)['arr_0']
m.scatters = np.load("%s/scatters.npz" %DATA_DIR)['arr_0']
m.chisqs = np.load("%s/chisqs.npz" %DATA_DIR)['arr_0']
m.pivots = np.load("%s/pivots.npz" %DATA_DIR)['arr_0']

m.infer_spectra(ds)

model = m.model_spectra[choose]
data = ds.test_flux[choose]

plt.plot(wl, model[0], c='r')
plt.plot(wl, data[0], c='k')
plt.show()

