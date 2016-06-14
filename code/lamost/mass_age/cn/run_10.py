"""
Training step for the paper: four labels + Ak + C + N
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import pyfits
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon/TheCannon')
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon')
from TheCannon import dataset
from TheCannon import model
from TheCannon import lamost
from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os


def load_data():
    a = pyfits.open("labels_file_full.fits")
    tbl = a[1].data
    a.close()
    
    # discard objects with unreliable C & N
    # Raw Teff greater than 4550 if -1 < [M/H] < -0.5
    # the most metal-poor stars [M/H] < -0.1 have sketchy scaling relations
    # but this shouldn't affect our spectral C and N 
    raw_teff = tbl['FPARAM'][:,0]
    mh = tbl['apogee_mh']
    choose_teff = raw_teff > 4550
    choose_mh = np.logical_and(-1 < mh, mh < -0.5)
    bad = np.logical_and(choose_mh, choose_teff) # 955 objects

    tr_id = tbl['lamost_id'][~bad]
    tr_id = np.array([val.strip() for val in tr_id])
    # teff, logg, mh, cm, nm, am, ak
    tr_label = np.hstack((tbl['PARAM'][:,0:2], tbl['PARAM'][:,3:], tbl['AK_WISE'][:,None]))[~bad]
    np.savez("./tr_id.npz", tr_id)
    np.savez("./tr_label.npz", tr_label)

    all_id = np.load("../run_2_train_on_good/all_ids.npz")['arr_0']
    all_flux = np.load("../run_2_train_on_good/test_flux.npz")['arr_0']
    all_ivar = np.load("../run_2_train_on_good/test_ivar.npz")['arr_0']
    good = np.array([np.where(all_id==f)[0][0] for f in tr_id])
    good_flux = all_flux[good,:]
    good_ivar = all_ivar[good,:]
    np.savez("tr_flux.npz", good_flux)
    np.savez("tr_ivar.npz", good_ivar)


def train():
    wl = np.load("wl.npz")['arr_0']
    tr_id = np.load("tr_id.npz")['arr_0']
    tr_label = np.load("tr_label.npz")['arr_0']
    tr_flux = np.load("tr_flux.npz")['arr_0']
    tr_ivar = np.load("tr_ivar.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label, tr_id, tr_flux, tr_ivar)
    # teff, logg, mh, cm, nm, am, ak
    ds.set_label_names(
            ['T_{eff}', '\log g', '[Fe/H]', '[C/M]','[N/M]', 
                '[\\alpha/M]', 'A_k'])
    ds.diagnostics_SNR()
    ds.diagnostics_ref_labels()
    np.savez("tr_snr.npz", ds.tr_SNR)

    m = model.CannonModel(2)
    m.fit(ds)
    np.savez("./coeffs.npz", m.coeffs)
    np.savez("./scatters.npz", m.scatters)
    np.savez("./chisqs.npz", m.chisqs)
    np.savez("./pivots.npz", m.pivots)
    m.diagnostics_leading_coeffs(ds)
    m.diagnostics_leading_coeffs_triangle(ds)
    m.diagnostics_plot_chisq(ds)


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def test_step():
    wl = np.load("./wl.npz")['arr_0']
    tr_id = np.load("./tr_id.npz")['arr_0']
    tr_flux = np.load("./tr_flux.npz")['arr_0']
    tr_ivar = np.load("./tr_ivar.npz")['arr_0']
    tr_label = np.load("./tr_label.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label, tr_id, tr_flux, tr_ivar)
    ds.set_label_names(
            ['T_{eff}', '\log g', '[Fe/H]', '[C/M]','[N/M]', 
                '[\\alpha/M]', 'A_k'])

    m = model.CannonModel(2)
    m.coeffs = np.load("./coeffs.npz")['arr_0']
    m.scatters = np.load("./scatters.npz")['arr_0']
    m.chisqs = np.load("./chisqs.npz")['arr_0']
    m.pivots = np.load("./pivots.npz")['arr_0']

    nguesses = 10
    nobj = len(tr_id)
    nlabels = len(m.pivots)
    choose = np.random.randint(0,nobj,size=nguesses)    
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
    ds.diagnostics_survey_labels()
    ds.diagnostics_1to1(figname = "1to1_test_label")


if __name__=="__main__":
    #load_data()
    #train()
    #print("test")
    test_step()
