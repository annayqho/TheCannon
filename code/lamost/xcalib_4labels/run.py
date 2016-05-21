"""
Run the test step on all the LAMOST DR2 objects.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho/xcalib_4labels/TheCannon')
sys.path.insert(0, 'home/annaho/xcalib_4labels')
#from lamost import load_spectra
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
    wl = np.load("wl.npz")['arr_0']
    test_ID = np.load("output/%s_ids.npz" %date)['arr_0']
    print(str(len(test_ID)) + " objects")
    test_flux = np.load("output/%s_norm.npz" %date)['arr_0']
    test_ivar = np.load("output/%s_norm.npz" %date)['arr_1']

    nlabels = 4
    nobj = len(test_ID)

    lamost_label = np.load("output/%s_tr_label.npz" %date)['arr_0']

    ds = dataset.Dataset(wl, test_ID, test_flux[0:2,:], test_ivar[0:2,:], lamost_label, 
            test_ID, test_flux, test_ivar)
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

    m = model.CannonModel(2)
    m.coeffs = np.load("./coeffs.npz")['arr_0']
    m.scatters = np.load("./scatters.npz")['arr_0']
    m.chisqs = np.load("./chisqs.npz")['arr_0']
    m.pivots = np.load("./pivots.npz")['arr_0']

    nguesses = 7
    starting_guesses = np.zeros((nguesses,nlabels)) 
    hiT_hiG_hiM = np.array([  5.15273730e+03,   3.71762228e+00,   3.16861898e-01, 2.46907920e-02])
    hiT_hiG_loM = np.array([  5.16350098e+03,   3.45917511e+00,  -9.24426436e-01, 2.49296919e-01])
    loT_loG_hiM = np.array([  4.04936841e+03,   1.47109437e+00,   2.07210138e-01, 1.49733415e-02])
    loT_loG_loM = np.array([  4.00651318e+03,   8.35013509e-01,  -8.98257852e-01, 7.65705928e-02])
    high_alpha = np.array([[4750, 2.6, -0.096, 0.25]])
    low_alpha = np.array([[4840, 2.67, -0.045, 0.049]])
    low_feh = np.array([[4500, 1.45, -1.54, 0.24]])
    starting_guesses[0,:] = hiT_hiG_hiM-m.pivots
    starting_guesses[1,:] = hiT_hiG_loM-m.pivots
    starting_guesses[2,:] = loT_loG_loM-m.pivots
    starting_guesses[3,:] = loT_loG_hiM-m.pivots
    starting_guesses[4,:] = high_alpha-m.pivots
    starting_guesses[5,:] = low_alpha-m.pivots
    starting_guesses[6,:] = low_feh-m.pivots

    labels = np.zeros((nguesses, nobj, nlabels)) # 4,10955,4
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
    ds.diagnostics_survey_labels(figname="%s_survey_labels_triangle.png" %date)
    ds.test_label_vals = best_labels[:,0:3]
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]'])
    ds.diagnostics_1to1(figname = "%s_1to1_test_label" %date)


if __name__=="__main__":
    dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])
    dates = np.delete(dates, np.where(dates=='20140330')[0][0]) # no obj
    dates = np.delete(dates, np.where(dates=='20121028')[0][0]) # no obj
    for date in dates:
        print("running %s" %date)
        if glob.glob("output/%s_all_cannon_labels.npz" %date): print("already done")
        else: 
            failures = ['20140118', '20120201', '20121017', '20120105', '20111129', '20111110', '20130307'] # fails to converge`
            if date not in failures: 
                test_step(date)
