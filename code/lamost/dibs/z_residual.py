""" plot spectral residuals, sorted by height above the Galactic plane """

from TheCannon.lamost import load_spectra
import sys
import pyfits
sys.path.append("/Users/annaho/Github/Spectra")
sys.path.append("/Users/annaho/Github/TheCannon/code/lamost/li_giants")
from astropy import units as u
from astropy.coordinates import SkyCoord
from normalize import normalize
from plot_residual import plot
import glob
import numpy as np
from TheCannon import model
from TheCannon import dataset
from matplotlib import rc
rc("text", usetex=True)
rc("font", family="serif")

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

inds = np.array([np.where(cat['LAMOST_ID']==val)[0][0] for val in ids])
ra = cat['RA'][inds]
dec = cat['Dec'][inds]
teff = cat['Teff'][inds]
logg = cat['logg'][inds]
mh = cat['FeH'][inds]
alpham = cat['alphaM'][inds]
ak = 0.05*np.ones(len(inds))
lab = np.vstack((teff,logg,mh,alpham,ak))

ds = dataset.Dataset(
        wl, ids, norm_flux, norm_ivar, lab, ids, norm_flux, norm_ivar)

ds.test_label_vals = lab.T

# generate model test spectra
m.infer_spectra(ds)

Cinv = ds.test_ivar / (1 + ds.test_ivar*m.scatters**2)
#res = Cinv*(ds.test_flux - m.model_spectra)**2
res = (ds.test_flux - m.model_spectra)

# get height above the plane
c = SkyCoord(ra, dec, unit='deg')
lat = np.abs(c.icrs.galactic.b)

for ii in range(0, len(ids)):
    prefix = ids[ii].split(".")[0]
    plot(ii, wl, norm_flux, norm_ivar, m.model_spectra, m.coeffs, m.scatters, m.chisq, m.pivots, 5600, 6400, [5780, 5797, 6283], "%s.png" %prefix) 
