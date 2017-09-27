""" plots of spectra around the H-alpha line for 1,000 LAMOST sub-giant stars
that fit Andy Casey's criteria """

from TheCannon.lamost import load_spectra
import sys
import pyfits
from plot_residual import plot
from residuals import load_model
sys.path.append("/Users/annaho/Github/Spectra")
from normalize import normalize
import glob
import numpy as np
from TheCannon import model
from TheCannon import dataset

# load
specdir = "/Users/annaho/Github/TheCannon/data/LAMOST/Li_Giants/Spectra_Random_1000"
files = np.array(glob.glob(specdir+"/*.fits"))
ids = []
for ii,val in enumerate(files):
    ids.append(val.split("/")[-1])
wl, flux, ivar = load_spectra(files)

# normalize
norm_flux, norm_ivar = normalize(wl, flux, ivar, L=50)

# import model parameters
modeldir = "/Users/annaho/Github/TheCannon/data/LAMOST/Label_Transfer"
chisq = np.load(modeldir + "/chisqs.npz")['arr_0']
coeff = np.load(modeldir + "/coeffs.npz")['arr_0']
scat = np.load(modeldir + "/scatters.npz")['arr_0']
pivot = np.load(modeldir + "/pivots.npz")['arr_0']

# initialize dataset and model
m = model.CannonModel(2, useErrors=False)
m.coeffs = coeff
m.chisq = chisq
m.scatters = scat
m.pivots = pivot
m.scales = np.ones(len(pivot))

# labels
labeldir = "/Users/annaho/Github/TheCannon/data/LAMOST/Label_Transfer"
inputf = pyfits.open("%s/Ho_et_all_catalog_resubmit.fits" %labeldir)
cat = inputf[1].data
inputf.close()

inds = np.array([np.where(cat['id']==val)[0][0] for val in ids])
teff = cat['cannon_teff'][inds]
logg = cat['cannon_logg'][inds]
mh = cat['cannon_m_h'][inds]
alpham = cat['cannon_alpha_m'][inds]
ak = 0.05*np.ones(len(inds))
lab = np.vstack((teff,logg,mh,alpham,ak))

ds = dataset.Dataset(
        wl, ids, norm_flux, norm_ivar, lab, ids, norm_flux, norm_ivar)

ds.test_label_vals = lab.T

# generate model test spectra
m.infer_spectra(ds)

for ii in range(0,len(ids)):
    prefix = ids[ii].split(".")[0]
    plot(ii, wl, norm_flux, norm_ivar, m.model_spectra, m.coeffs, m.scatters, m.chisq, m.pivots, "%s.png" %prefix) 
