import numpy as np
from astropy.table import Table
from TheCannon.lamost import load_spectra
import matplotlib.pyplot as plt
from TheCannon import dataset
from TheCannon import model

labdir = "/Users/annaho/Github_Repositories/TheCannon/docs/source"
data = Table.read("%s/lamost_labels.fits" %labdir)
print(data.colnames)
filename = data['LAMOST_ID'][0].strip()
print(filename)

specdir = "/Users/annaho/Github_Repositories/TheCannon/docs/source/spectra"
wl, flux, ivar = load_spectra("%s/" %specdir + filename)

plt.step(wl, flux, where='mid', linewidth=0.5, color='k')
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.savefig("sample_spec.png")
plt.close()
# Now, do all of the files
filenames = np.array([val.strip() for val in data['LAMOST_ID']])
filenames_full = np.array([specdir+"/"+val.strip() for val in filenames])
wl, flux, ivar = load_spectra(filenames_full)

#plt.hist(SNR)
#plt.show()

# Use 1000 stars as a training set
tr_flux = flux[0:1000]
tr_ivar = ivar[0:1000]
tr_ID = filenames[0:1000]

# Now you need to get the training labels. 

inds = np.array([np.where(filenames==val)[0][0] for val in tr_ID])
tr_teff = data['TEFF'][inds]
tr_logg = data['LOGG'][inds]
tr_mh = data['PARAM_M_H'][inds]
tr_alpham = data['PARAM_ALPHA_M'][inds]

# Take a look at the teff-logg diagram, color-coded by metallicity
plt.scatter(tr_teff, tr_logg, c=tr_mh, lw=0, s=7, cmap="viridis")
plt.gca().invert_xaxis()
plt.xlabel("Teff")
plt.ylabel("logg")
plt.colorbar(label="M/H")
plt.savefig("teff_logg_training.png")
plt.close()

# Note that there are very few stars at low metallicity,
# so it will probably be challenging to do as good of a job
# or get as precise results here.

# OK, so now we need to get our stuff into the format specified on the website

# Wavelength grid with shape [num_pixels]
print(wl.shape)

# So, there are 3626 pixels.

# Block of flux values, inverse variance values with shape
print(tr_ID.shape)
print(tr_flux.shape)
print(tr_ivar.shape)
# [num_training_objects, num_pixels]
# (1339, 3626)
# Fine. Not normalized yet, but we will do that later.

# Now we need a block of training labels
# [num_training_objects, num_labels]
# Right now we have them separate, combine into an array of this shape:

tr_label = np.vstack((tr_teff, tr_logg, tr_mh, tr_alpham))
# Note that that gives us (4,1339) which is (num_labels, num_tr_obj)
# So we need to take the transpose

tr_label = np.vstack((tr_teff, tr_logg, tr_mh, tr_alpham)).T

# Now we need to define our "test set": a bunch of other
# spectra whose labels we want to determine and don't know yet.
# Let's use some of the other spectra in the dataset
# Say, the ones with 80 < SNR < 100
test_ID = filenames[1000:]
test_flux = flux[1000:]
test_ivar = ivar[1000:]

# Check the sizes
print(test_ID.shape)
print(test_flux.shape)
print(test_ivar.shape)

# OK, excellent. Now we're ready!
ds = dataset.Dataset(
        wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[alpha/M]'])
fig = ds.diagnostics_SNR()
plt.savefig("SNR_hist.png")
plt.close()
fig = ds.diagnostics_ref_labels()
plt.savefig("ref_labels.png")
plt.close()

ds.continuum_normalize_gaussian_smoothing(L=50)
plt.step(ds.wl, ds.tr_flux[0], where='mid', linewidth=0.5, color='k')
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.savefig("norm_spec.png")
plt.close()

# Now, fit for a polynomial model of order 2 (quadratic)
m = model.CannonModel(2, useErrors=False)
m.fit(ds)
fig = m.diagnostics_leading_coeffs(ds)
plt.savefig("leading_coeffs.png")
plt.close()

# Test
print(ds.tr_label.shape)
print(m.pivots)
starting_guess = np.mean(ds.tr_label,axis=0)-m.pivots
print(starting_guess)

errs, chisq = m.infer_labels(ds, starting_guess)

ds.diagnostics_survey_labels()
plt.savefig("survey_labels.png")
plt.close()

inds = np.array([np.where(filenames==val)[0][0] for val in ds.test_ID])
test_teff = data['TEFF'][inds]
test_logg = data['LOGG'][inds]
test_mh = data['PARAM_M_H'][inds]
test_alpham = data['PARAM_ALPHA_M'][inds]
test_label = np.vstack((test_teff, test_logg, test_mh, test_alpham)).T
ds.tr_label = test_label

ds.diagnostics_1to1()
plt.savefig("survey_ref_comparison.png")
plt.close()
