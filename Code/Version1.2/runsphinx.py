# Compile the Sphinx documentation all at once

import numpy as np
readin = "reference_labels.txt"
IDs = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
filenames1 = []
for i in range(0, len(IDs)): #incorporate file location info
    filename = '/home/annaho/AnnaCannon/Data/APOGEE_Data' + IDs[i][1:]
    filenames1.append(filename)

from read_apogee import get_spectra
lambdas, norm_fluxes, norm_ivars, SNRs = get_spectra(filenames1)

from read_labels import get_reference_labels
IDs, all_label_names, all_label_values = get_reference_labels(readin)

from dataset import Dataset
reference_set = Dataset(IDs=IDs, SNRs=SNRs, lams=lambdas, fluxes = norm_fluxes, 
        ivars = norm_ivars, label_names=all_label_names, 
        label_vals=all_label_values)

cols = [1, 3, 5]
reference_set.choose_labels(cols)

Teff = reference_set.label_vals[:,0]
Teff_corr = all_label_values[:,2]
diff_t = np.abs(Teff-Teff_corr)
diff_t_cut = 600.
logg = reference_set.label_vals[:,1]
logg_cut = 100.
mask = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
reference_set.choose_objects(mask)

test_set = Dataset(IDs=reference_set.IDs, SNRs=reference_set.SNRs,
        lams=lambdas, fluxes=reference_set.fluxes, ivars = reference_set.ivars, 
        label_names=reference_set.label_names)

from dataset import dataset_prediagnostics
dataset_prediagnostics(reference_set, test_set)

from cannon1_train_model import train_model
model = train_model(reference_set)

from cannon1_train_model import model_diagnostics
model_diagnostics(reference_set, model)

from cannon2_infer_labels import infer_labels
test_set, covs = infer_labels(model, test_set)

from dataset import dataset_postdiagnostics
dataset_postdiagnostics(reference_set, test_set)

from spectral_model import draw_spectra
cannon_set = draw_spectra(model, test_set)

from spectral_model import diagnostics
diagnostics(cannon_set, test_set, model)
