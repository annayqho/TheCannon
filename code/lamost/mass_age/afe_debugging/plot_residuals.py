import numpy as np
import pyfits
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/annaho/Dropbox/Research/TheCannon/code/lamost/mass_age/paper_plots")
#sys.path.append("/Users/annaho/Data/LAMOST/Mass_And_Age")
from model_spectra import spectral_model
from TheCannon import dataset
from TheCannon import model

""" Plot residuals of stars with large alpha-enhancement
scatter and high (or low) radial velocities """

DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"

# Load training data
wl = np.load("%s/wl.npz" %DIR)['arr_0']
lamost_id = np.load("%s/ref_id.npz" %DIR)['arr_0']
ref_flux = np.load("%s/ref_flux.npz" %DIR)['arr_0']
ref_ivar = np.load("%s/ref_ivar.npz" %DIR)['arr_0']
ref_label = np.load("%s/ref_label.npz" %DIR)['arr_0']
cannon_label = np.load("%s/xval_cannon_label_vals.npz" %DIR)['arr_0']
snr = np.load("%s/ref_snr.npz" %DIR)['arr_0']
rv = np.load("%s/ref_rvs.npz" %DIR)['arr_0']
chisq = np.load("%s/xval_cannon_label_chisq.npz" %DIR)['arr_0']
coeffs = np.load("%s/coeffs.npz" %DIR)['arr_0']
pivots = np.load("%s/pivots.npz" %DIR)['arr_0']
scatters = np.load("%s/scatters.npz" %DIR)['arr_0']

# Create model spectra
m = model.CannonModel(2)
m.coeffs = coeffs
m.pivots = pivots
m.scatters = scatters
ds = dataset.Dataset([], [], [], [], [], [], ref_flux, ref_ivar)
ds.test_label_vals = cannon_label
m.infer_spectra(ds)

# Plot residuals of stars with large diff and large negative rvel
# (Neg rvel seems worse to me, in the scatterplot...)
diff = ref_label[:,5] - cannon_label[:,5]
choose_bad = np.logical_and(np.abs(diff) < 0.01, np.abs(rv) < 10)
choose_bad = np.logical_and(diff < -0.05, rv < -50)
choose_bad = np.logical_and(diff > 0.05, rv > 50)
choose_snr = snr > 70
#choose_chisq = np.logical_and(chisq > 1000, chisq < 10000)
#choose_quality = np.logical_and(choose_snr, choose_chisq)
#choose = np.logical_and(choose_bad, choose_quality)
choose = np.logical_and(choose_bad, choose_snr)

# Stack residuals
resid = ref_flux - m.model_spectra
stack = np.sum(resid[choose], axis=0)
median = np.median(resid[choose], axis=0)

# xmin = 5600
# xmax = 5900 
# val = lamost_id[choose][0]
# fig = spectral_model(val, xmin=xmin, xmax=xmax)
# plt.show()

# for val in lamost_id[choose][0:5]:
#      resid, fig = spectral_model(val, xmin=xmin, xmax=xmax)
     #plt.savefig("%s_DIB_5780.png" %(val))#, xmin, xmax))
# 
#plt.scatter(rv, diff, lw=0, c='k')
#plt.scatter(rv[choose], diff[choose], lw=0, c='r')
#plt.show()

