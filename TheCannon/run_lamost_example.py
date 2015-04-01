from __future__ import (absolute_import, division, print_function)
from lamost import LamostDataset
from cannon.model import CannonModel
import numpy as np

###### WORKFLOW

# RUN APOGEE MUNGING CODE
dataset = LamostDataset("example_LAMOST/Training_Data",
                        "example_LAMOST/Training_Data",
                        "example_DR12/reference_labels.csv")

# Choose labels
cols = ['teff', 'logg', 'feh']
dataset.choose_labels(cols)

# set the headers for plotting
dataset.set_label_names_tex(['T_{eff}', '\log g', '[M/H]'])

# Plot SNR distributions and triangle plot of reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()

# Pseudo-continuum normalization
dataset.continuum_normalize(q=0.90, delta_lambda=50)
pseudo_cont_dataset = dataset
dataset = LamostDataset("example_LAMOST/Testing",
                        "example_LAMOST/Testing",
                        "example_DR12/reference_labels.csv")

# RUN CONTINUUM IDENTIFICATION CODE
start = 300
end = 400
contmask = np.ma.array(contmask, mask=bad)
# make sure dataset here is pseudo_cont_norm dataset
f_bar = np.zeros(len(dataset.wl))
sigma_f = np.zeros(len(dataset.wl))
nbad = np.zeros(len(dataset.wl))
for wl in range(0,len(dataset.wl)):
    array = dataset.tr_fluxes[:,wl]
    f_bar[wl] = np.median(array[array>0])
    nbad[wl] = sum(array==0)
    ngood = len(array==0)-sum(array==0)
    sigma_f[wl] = 100*np.var(array[array>0])
    #sigma_f[wl] = 100*np.var(array[array>0])/(2*(ngood))**0.5
    nbad[wl] = sum(array==0)

bad = np.var(dataset.tr_ivars, axis=0) == 0
f_bar = np.ma.array(f_bar, mask=bad)
sigma_f = np.ma.array(sigma_f, mask=bad)

start = 0 
end = len(dataset.wl)

f,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(dataset.wl[start:end], f_bar[start:end], alpha=0.5)
axarr[0].fill_between(dataset.wl[start:end], (f_bar+sigma_f)[start:end], (f_bar-sigma_f)[start:end], alpha=0.2)
axarr[0].set_ylabel("Med & Var Flux Across Stars")
axarr[1].plot(dataset.wl[start:end], nbad[start:end], alpha=0.7)
axarr[1].set_ylabel("# Bad Stars")
axarr[1].set_xlabel("Wavelength (A)")
axarr[0].set_title("Pixel Consistency Across Stars")

# the number of bad pixels in each star
ax.imshow(dataset.tr_fluxes)

f_bar_x = np.median(dataset.tr_fluxes[:,start:end], axis=0)
sigma_f_x = np.var(dataset.tr_fluxes[:,start:end], axis=0)
plot(sigma_f)
plot(f_bar)
bad = f_bar == 0
f_bar = np.ma.array(f_bar, mask=bad)
sigma_f = np.ma.array(sigma_f, mask=bad)
npix = len(bad)-sum(bad)
np.ma.mean(sigma_f) 
0.15
np.ma.mean(f_bar) 
0.87
scatter(sigma_f, np.abs(f_bar-1))

f_cut = 0.15 
sig_cut = 0.16

contmask = np.zeros(end-start, dtype=bool)
contmask[np.logical_and(np.abs(f_bar-1) <= f_cut, sigma_f <= sig_cut)] = True
contmask = np.ma.array(contmask, mask=bad)
contmask = np.ma.filled(contmask, fill_value=False)
sum(contmask) # looking for ~60-100...

plot(dataset.wl[start:end], f_bar, alpha=0.7)
scatter(dataset.wl[start:end][contmask], f_bar[contmask], c='r')
scatter(sigma_f[contmask], np.abs(1-f_bar[contmask]))
np.where(contmask==True)

pseudo_cont_dataset.find_continuum(f_cut=0.003, sig_cut=0.09)
dataset.set_continuum(pseudo_cont_dataset.contmask)

dataset.ranges = [[0,1883],[2094,3899]]

tr_cont, test_cont = dataset.fit_continuum(deg=5)


# RUN CONTINUUM NORMALIZATION CODE
dataset.ranges = None
dataset.continuum_normalize(cont=(tr_cont, test_cont))

# learn the model from the reference_set
model = CannonModel(dataset, 2) # 2 = quadratic model
model.fit() # model.train would work equivalently.

# check the model
model.diagnostics()

# infer labels with the new model for the test_set
dataset, label_errs = model.infer_labels(dataset)
#dataset, covs = model.predict(dataset)

# Make plots
dataset.dataset_postdiagnostics(dataset)

cannon_set = model.draw_spectra(dataset)
model.spectral_diagnostics(dataset)
