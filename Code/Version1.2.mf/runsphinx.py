"""
Compile the Sphinx documentation all at once
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from apogee.read_apogee import get_spectra
from apogee.read_labels import get_reference_labels

from cannon.dataset import (Dataset, dataset_prediagnostics,
                            dataset_postdiagnostics)
from cannon.cannon1_train_model import (train_model, model_diagnostics)
from cannon.cannon2_infer_labels import infer_labels
from cannon.spectral_model import (draw_spectra, diagnostics)


lambdas, norm_fluxes, norm_ivars, SNRs = get_spectra("example_DR10/Data")

IDs, all_label_names, all_label_values = \
    get_reference_labels("example_DR10/reference_labels_update.txt")

reference_set = Dataset(IDs=IDs, SNRs=SNRs, lams=lambdas, fluxes=norm_fluxes,
                        ivars=norm_ivars, label_names=all_label_names,
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

test_set = Dataset(IDs=reference_set.IDs, SNRs=reference_set.SNRs, lams=lambdas,
                   fluxes=reference_set.fluxes, ivars=reference_set.ivars,
                   label_names=reference_set.label_names)

dataset_prediagnostics(reference_set, test_set)

model = train_model(reference_set)

model_diagnostics(reference_set, model)

test_set, covs = infer_labels(model, test_set)

dataset_postdiagnostics(reference_set, test_set)

cannon_set = draw_spectra(model, test_set)

diagnostics(cannon_set, test_set, model)
