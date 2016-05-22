"""
In this run, I train TC on all *good* objects in the 11,057 overlap set 
(this is the 9594 objects from run_2, + the 500-something objects in 
examples/example_DR12/temp_keep_metalpoor.txt) minus the 4 ish objects
that havebad AKWISE values. The label file here was made using topcat
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon/TheCannon')
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon')
from lamost import load_spectra
from TheCannon import dataset
from TheCannon import model
from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os
import pyfits


def train():
    a = pyfits.open("label_file.fits") 
    tbdata = a[1].data
    a.close()
    apogee_teff = tbdata['apogee_teff']
    apogee_logg = tbdata['apogee_logg']
    apogee_mh = tbdata['apogee_mh']
    apogee_alpham = tbdata['apogee_alpham']
    apogee_reddening = tbdata['AK_WISE']
    tr_label = np.vstack((apogee_teff,apogee_logg,apogee_mh,apogee_alpham,apogee_reddening)).T
    tr_id_full = tbdata['lamost_id']
    tr_id = np.array([val.strip() for val in tr_id_full])

    all_id = np.load("../run_2_train_on_good/all_ids.npz")['arr_0']
    all_flux = np.load("../run_2_train_on_good/test_flux.npz")['arr_0']
    all_ivar = np.load("../run_2_train_on_good/test_ivar.npz")['arr_0']
    good = np.array([np.where(all_id==f)[0][0] for f in tr_id])

    good_flux = all_flux[good,:] 
    good_ivar = all_ivar[good,:]

    np.savez("tr_id.npz", tr_id)
    np.savez("tr_label.npz", tr_label)
    np.savez("tr_flux.npz", good_flux)
    np.savez("tr_ivar.npz", good_ivar)

    wl = np.load("../run_2_train_on_good/wl.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, good_flux, good_ivar, tr_label, tr_id, good_flux, good_ivar)
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]', 'AKWISE'])
    ds.diagnostics_SNR()
    #ds.diagnostics_ref_labels()
    np.savez("tr_snr.npz", ds.tr_SNR)

    m = model.CannonModel(2)
    m.fit(ds)
    np.savez("./coeffs.npz", m.coeffs)
    np.savez("./scatters.npz", m.scatters)
    np.savez("./chisqs.npz", m.chisqs)
    np.savez("./pivots.npz", m.pivots)
    m.diagnostics_leading_coeffs(ds)
    #m.diagnostics_leading_coeffs_triangle(ds)
    m.diagnostics_plot_chisq(ds)


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def test_step():
    wl = np.load("../run_2_train_on_good/wl.npz")['arr_0']
    direc = "../../examples/test_small_random"
    tr_id = np.load("./tr_id.npz")['arr_0']
    tr_flux = np.load("./tr_flux.npz")['arr_0']
    tr_ivar = np.load("./tr_ivar.npz")['arr_0']
    tr_label = np.load("./tr_label.npz")['arr_0']

    ds = dataset.Dataset(wl, tr_id, tr_flux, tr_ivar, tr_label, tr_id, tr_flux, tr_ivar)
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]', 'AKWISE'])

    m = model.CannonModel(2)
    m.coeffs = np.load("./coeffs.npz")['arr_0']
    m.scatters = np.load("./scatters.npz")['arr_0']
    m.chisqs = np.load("./chisqs.npz")['arr_0']
    m.pivots = np.load("./pivots.npz")['arr_0']

    nguesses = 7
    nobj = len(tr_id)
    nlabels = len(m.pivots)
    choose = np.random.randint(0,nobj,size=nguesses)
    apogee_label = np.load("./tr_label.npz")['arr_0']
    starting_guesses = tr_label[choose]-m.pivots
    labels = np.zeros((nguesses, nobj, nlabels))
    chisq = np.zeros((nguesses, nobj))
    errs = np.zeros(labels.shape)

    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,m,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    np.savez("labels_all_starting_vals.npz", labels)
    np.savez("chisq_all_starting_vals.npz", chisq)
    np.savez("errs_all_starting_vals.npz", errs)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros(tr_label.shape)
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("./all_cannon_labels.npz", best_labels)
    np.savez("./cannon_label_chisq.npz", best_chisq)
    np.savez("./cannon_label_errs.npz", best_errs)

    ds.test_label_vals = best_labels
    #ds.diagnostics_survey_labels()
    ds.diagnostics_1to1(figname = "1to1_test_label")


if __name__=="__main__":
    #train()
    test_step()
