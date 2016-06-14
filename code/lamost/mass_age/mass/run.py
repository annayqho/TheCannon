"""
Run the test step on all the LAMOST DR2 objects.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho')
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


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def test_step(date):
    direc = "../xcalib_4labels"
    wl = np.load("%s/wl.npz" %direc)['arr_0']
    test_ID = np.load("%s/output/%s_ids.npz" %(direc, date))['arr_0']
    print(str(len(test_ID)) + " objects")
    test_flux = np.load("%s/output/%s_norm.npz" %(direc,date))['arr_0']
    test_ivar = np.load("%s/output/%s_norm.npz" %(direc,date))['arr_1']

    lamost_label = np.load("%s/output/%s_tr_label.npz" %(direc,date))['arr_0']
    apogee_label = np.load("./tr_label.npz")['arr_0']

    ds = dataset.Dataset(wl, test_ID, test_flux[0:2,:], test_ivar[0:2,:], 
            lamost_label, test_ID, test_flux, test_ivar)
    ds.set_label_names(
            ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]', 'log M', 'A_k'])

    m = model.CannonModel(2)
    m.coeffs = np.load("./coeffs.npz")['arr_0']
    m.scatters = np.load("./scatters.npz")['arr_0']
    m.chisqs = np.load("./chisqs.npz")['arr_0']
    m.pivots = np.load("./pivots.npz")['arr_0']

    nlabels = len(m.pivots)
    nobj = len(test_ID)

    nguesses = 7
    choose = np.random.randint(0,apogee_label.shape[0],size=nguesses)
    starting_guesses = apogee_label[choose,:]-m.pivots

    labels = np.zeros((nguesses, nobj, nlabels)) 
    chisq = np.zeros((nguesses, nobj))
    errs = np.zeros(labels.shape)
    
    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,m,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    np.savez("output/%s_cannon_label_guesses.npz" %date, labels)
    np.savez("output/%s_cannon_chisq_guesses.npz" %date, labels)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("output/%s_all_cannon_labels.npz" %date, best_labels)
    np.savez("output/%s_cannon_label_chisq.npz" %date, best_chisq)
    np.savez("output/%s_cannon_label_errs.npz" %date, best_errs)

    ds.test_label_vals = best_labels
    #ds.diagnostics_survey_labels(figname="%s_survey_labels_triangle.png" %date)
    ds.test_label_vals = best_labels[:,0:3]
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]'])
    ds.diagnostics_1to1(figname = "%s_1to1_test_label" %date)


if __name__=="__main__":
    dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])
    for date in dates:
        print("running %s" %date)
        if glob.glob("output/%s_all_cannon_labels.npz" %date): print("already done")
        else: 
            test_step(date)
