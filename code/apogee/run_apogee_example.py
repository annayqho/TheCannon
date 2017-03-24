import numpy as np
import glob
from TheCannon import apogee
from TheCannon import dataset
from TheCannon import model 

# (1) PREPARE DATA

tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("example_DR10/Data")
tr_label = apogee.load_labels("example_DR10/reference_labels.csv")

# doing a 1-to-1 test for simplicity
test_ID = tr_ID
test_flux = tr_flux 
test_ivar = tr_ivar
tr_label = apogee.load_labels("example_DR10/reference_labels.csv")

# choose labels and make a new array 
ds = dataset.Dataset(
        wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

# set LaTeX label names for making diagnostic plots
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

# Plot SNR distributions and triangle plot of reference labels
fig = ds.diagnostics_SNR()
fig = ds.diagnostics_ref_labels()

# (2) IDENTIFY CONTINUUM PIXELS
pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(
        q=0.90, delta_lambda=50)

ds.ranges = [[371,3192], [3697,5500], [5500,5997], [6461,8255]]
contmask = ds.make_contmask(
        pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)

ds.set_continuum(contmask)
cont = ds.fit_continuum(3, "sinusoid")

norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
        ds.continuum_normalize(cont)

ds.tr_flux = norm_tr_flux
ds.tr_ivar = norm_tr_ivar
ds.test_flux = norm_test_flux
ds.test_ivar = norm_test_ivar

from TheCannon import model
md = model.CannonModel(2)
md.fit(ds)
md.diagnostics_contpix(ds)
md.diagnostics_leading_coeffs(ds)
md.diagnostics_plot_chisq(ds)

label_errs = md.infer_labels(ds)
test_labels = ds.test_label_vals
ds.diagnostics_test_step_flagstars()
ds.diagnostics_survey_labels()
dset.diagnostics_1to1()
