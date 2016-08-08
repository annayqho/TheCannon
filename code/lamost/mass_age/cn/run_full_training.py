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

GIT_DIR = "/Users/annaho/Dropbox/Research/TheCannon/"
DATA_DIR = "data/"
SPEC_DIR = "/Users/annaho/Data/LAMOST"

def load_data():
    print("Loading all data")
    DIR = GIT_DIR + DATA_DIR
    a = pyfits.open("%s/labels_file_full.fits" %DIR)
    tbl = a[1].data
    a.close()

    # Pull out all APOGEE DR12 values
    # FPARAM: (teff, logg, rvel, mh, c, n, alpha)
    teff_all = tbl['FPARAM'][:,0]
    logg_all = tbl['FPARAM'][:,1]
    mh_all = tbl['FPARAM'][:,3]
    cm_all = tbl['FPARAM'][:,4]
    nm_all = tbl['FPARAM'][:,5]
    am_all = tbl['FPARAM'][:,6]
    ak_all = tbl['AK_WISE']

    # Discard objects with Teff > 4550 if -1 < [M/H] < -0.5
    print("Discarding objects")
    choose_teff = teff_all > 4550
    choose_mh = np.logical_and(-1 < mh_all, mh_all < -0.5)
    discard_teff = np.logical_and(choose_mh, choose_teff) # 955 objects

    # Discard objects with [C/M] < -0.4 dex
    discard_cm = cm_all < -0.4

    # metal-poor stars [M/H] < -0.1 have sketchy scaling relations
    # but this shouldn't affect our spectral C and N 
    # in Marie's paper they don't have any low-metallicity stars,
    # but it doesn't matter for the training anyway.
    bad = np.logical_and(discard_teff, discard_cm)
    choose = ~bad

    ref_id = tbl['lamost_id'][choose]
    ref_id = np.array([val.strip() for val in ref_id]).astype(str)
    ref_label = np.hstack((
            teff_all[choose], logg_all[choose], mh_all[choose],
            cm_all[choose], nm_all[choose], am_all[choose], 
            ak_all[choose]))

    np.savez("./ref_id.npz", ref_id)
    np.savez("./ref_label.npz", ref_label)

    print("Getting spectra")
    all_id = np.load("%s/tr_id.npz" %SPEC_DIR)['arr_0'].astype(str)
    all_flux = np.load("%s/tr_flux.npz" %SPEC_DIR)['arr_0']
    all_ivar = np.load("%s/tr_ivar.npz" %SPEC_DIR)['arr_0']
    choose = np.array([np.where(all_id==f)[0][0] for f in ref_id])
    flux = all_flux[choose,:]
    ivar = all_ivar[choose,:]
    np.savez("ref_flux.npz", flux)
    np.savez("ref_ivar.npz", ivar)


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
    load_data()
    #train()
    #print("test")
    #test_step()
