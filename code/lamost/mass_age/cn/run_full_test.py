"""
Run the test step on all the LAMOST DR2 objects.
You have to run this script on aida42082
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho/TheCannon')
#sys.path.insert(0, '/home/annaho')
#from lamost import load_spectra
#import dataset
#import model
from TheCannon import dataset
from TheCannon import model
#from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os
from pull_data import find_colors, apply_mask

SPEC_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels"
COL_DIR = "/home/annaho/TheCannon/data/lamost"
MODEL_DIR = "."


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def test_step(date):
    wl = np.load("%s/wl_cols.npz" %MODEL_DIR)['arr_0']
    test_ID = np.load("%s/output/%s_ids.npz" %(SPEC_DIR, date))['arr_0']
    print(str(len(test_ID)) + " objects")
    test_flux_temp = np.load("%s/output/%s_norm.npz" %(SPEC_DIR,date))['arr_0']
    test_ivar_temp = np.load("%s/output/%s_norm.npz" %(SPEC_DIR,date))['arr_1']

    # Mask
    mask = np.load("mask.npz")['arr_0']
    test_ivar_masked = apply_mask(wl[0:3626], test_ivar_temp, mask)

    # Append colors
    col = np.load(COL_DIR + "/" + date + "_col.npz")['arr_0']
    col_ivar = np.load(COL_DIR + "/" + date + "_col_ivar.npz")['arr_0']
    bad_flux = np.logical_or(np.isnan(col), col==np.inf)
    col[bad_flux] = 1.0
    col_ivar[bad_flux] = 0.0
    bad_ivar = np.logical_or(np.isnan(col_ivar), col_ivar==np.inf)
    col_ivar[bad_ivar] = 0.0
    test_flux = np.hstack((test_flux_temp, col.T))
    test_ivar = np.hstack((test_ivar_temp, col_ivar.T))
    
    lamost_label = np.load("%s/output/%s_tr_label.npz" %(SPEC_DIR,date))['arr_0']
    apogee_label = np.load("./ref_label.npz")['arr_0']

    ds = dataset.Dataset(wl, test_ID, test_flux[0:2,:], test_ivar[0:2,:], 
            lamost_label, test_ID, test_flux, test_ivar)

    np.savez(COL_DIR + "/test_flux.npz", ds.test_flux)
    np.savez(COL_DIR + "/test_ivar.npz", ds.test_ivar)
    np.savez(COL_DIR + "/test_snr.npz", ds.test_SNR)

    ds.set_label_names(
            ['T_{eff}', '\log g', '[Fe/H]', '[C/M]', '[N/M]', '[\\alpha/Fe]', 'A_k'])

    m = model.CannonModel(2)
    m.coeffs = np.load("./coeffs.npz")['arr_0']
    m.scatters = np.load("./scatters.npz")['arr_0']
    m.chisqs = np.load("./chisqs.npz")['arr_0']
    m.pivots = np.load("./pivots.npz")['arr_0']

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

    np.savez(COL_DIR + "/%s_cannon_label_guesses.npz" %date, labels)
    np.savez(COL_DIR + "/%s_cannon_chisq_guesses.npz" %date, labels)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez(COL_DIR + "/%s_all_cannon_labels.npz" %date, best_labels)
    np.savez(COL_DIR + "/%s_cannon_label_chisq.npz" %date, best_chisq)
    np.savez(COL_DIR + "/%s_cannon_label_errs.npz" %date, best_errs)

    ds.test_label_vals = best_labels
    #ds.diagnostics_survey_labels(figname="%s_survey_labels_triangle.png" %date)
    ds.test_label_vals = best_labels[:,0:3]
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]'])
    ds.diagnostics_1to1(figname = COL_DIR + "/%s_1to1_test_label" %date)


if __name__=="__main__":
    dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])
    for date in dates:
        print("running %s" %date)
        if glob.glob(COL_DIR + "/%s_all_cannon_labels.npz" %date): 
            print("already done")
        else: 
            test_step(date)
