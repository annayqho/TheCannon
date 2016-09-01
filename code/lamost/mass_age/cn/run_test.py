"""
The training set was a subset of the 8472 original objects
(because you did some culling).
Test on the ones you left out.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
from TheCannon import dataset
from TheCannon import model
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os

SPEC_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask"
MODEL_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/training_step"


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def test_step():
    wl = np.load(SPEC_DIR + "/wl_cols.npz")
    ref_id_all = np.load(SPEC_DIR + "/ref_id_col.npz")['arr_0']
    excised = np.load(SPEC_DIR + "/excised_obj/excised_ids.npz")['arr_0']
    inds = np.array([np.where(ref_id_all==val)[0][0] for val in excised])
    test_ID = ref_id_all[inds]
    print(str(len(test_ID)) + " objects")
    test_flux = np.load("%s/ref_flux_col.npz" %(SPEC_DIR))['arr_0'][inds]
    test_ivar = np.load("%s/ref_ivar_col.npz" %(SPEC_DIR))['arr_0'][inds]

    apogee_label = np.load("%s/ref_label.npz" %(SPEC_DIR))['arr_0'][inds]
    #np.savez("excised_label.npz", apogee_label)

    ds = dataset.Dataset(wl, test_ID, test_flux[0:2,:], test_ivar[0:2,:], 
            apogee_label, test_ID, test_flux, test_ivar)
    ds.set_label_names(
            ['T_{eff}', '\log g', '[Fe/H]', '[C/M]', '[N/M]', '[\\alpha/Fe]', 'A_k'])
    np.savez("excised_snr.npz", ds.test_SNR)
    print("DONE")

    m = model.CannonModel(2)
    m.coeffs = np.load(MODEL_DIR + "/coeffs.npz")['arr_0']
    m.scatters = np.load(MODEL_DIR + "/scatters.npz")['arr_0']
    m.chisqs = np.load(MODEL_DIR + "/chisqs.npz")['arr_0']
    m.pivots = np.load(MODEL_DIR + "/pivots.npz")['arr_0']

    nlabels = len(m.pivots)
    nobj = len(test_ID)

    nguesses = 7
    choose = np.random.randint(0,nobj,size=nguesses)
    print(apogee_label.shape)
    print(choose.shape)
    print(m.pivots.shape)
    starting_guesses = apogee_label[choose]-m.pivots

    labels = np.zeros((nguesses, nobj, nlabels)) 
    chisq = np.zeros((nguesses, nobj))
    errs = np.zeros(labels.shape)
    
    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,m,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("excised_all_cannon_labels.npz", best_labels)
    np.savez("excised_cannon_label_chisq.npz", best_chisq)
    np.savez("excised_cannon_label_errs.npz", best_errs)

    ds.test_label_vals = best_labels
    ds.diagnostics_1to1(figname = "excised_1to1_test_label")


if __name__=="__main__":
    test_step()
