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

DATA_DIR = "/Users/annaho/Data/LAMOST"

snr = np.load("%s/tr_snr.npz" %DATA_DIR)['arr_0']
chisq = np.load("%s/cannon_label_chisq.npz" %DATA_DIR)['arr_0']
choose = np.logical_and(snr > 100, chisq < 4000)
print(sum(choose))

wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
test_label = np.load("%s/all_cannon_labels.npz" %DATA_DIR)['arr_0']
ds.test_label_vals = test_label
tr_flux = np.load("%s/tr_flux.npz" %DATA_DIR)['arr_0']
tr_ivar = np.load("%s/tr_ivar.npz" %DATA_DIR)['arr_0']
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
resid = data-model

#for ii in range(len(resid)):
for ii in range(1120,1121):
    plt.plot(wl, resid[ii])
    plt.xlim(6400,7000)
    plt.ylim(-0.1,0.1)
    plt.axvline(x=6707.4, c='r', linestyle='--', linewidth=2)
    plt.axvline(x=6103, c='r', linestyle='--', linewidth=2)
    plt.savefig("resid_%s.png" %ii)
    #plt.close()
