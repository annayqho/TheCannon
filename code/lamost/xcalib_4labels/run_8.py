"""
Run the test step on all the LAMOST DR2 objects.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho/')
sys.path.insert(0, '/home/annaho/TheCannon')
from lamost import load_spectra
from TheCannon import dataset
from TheCannon import model
from lamost import load_spectra
#from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os


def prep_data(date):
    dir_files = "/home/annaho/xcalib_4labels/test_obj" 
    dir_dat = "/home/share/LAMOST/DR2/DR2_release/"
    test_ID = np.loadtxt("%s/%s_test_obj.txt" %(dir_files, date), dtype=str)
    print("%s obj" %len(test_ID))
    np.savez("output/%s_ids.npz" %date, test_ID)
    test_ID_long = np.array([dir_dat + f for f in test_ID])
    wl, test_flux, test_ivar, npix, SNRs = load_spectra(test_ID_long)
    np.savez("output/%s_SNRs.npz" %date, SNRs)
    np.savez("output/%s_frac_good_pix.npz" %date, npix)

    lamost_info = np.load("lamost_labels/lamost_labels_%s.npz" %date)['arr_0']
    inds = np.array([np.where(lamost_info[:,0]==a)[0][0] for a in test_ID])
    nstars = len(test_ID)
    lamost_info_sorted = np.zeros((nstars,4))
    lamost_label = lamost_info[inds,:][:,1:].astype(float)
    lamost_info_sorted[:,0:3] = lamost_label
    np.savez("output/%s_tr_label" %date, lamost_label)

    ds = dataset.Dataset(wl, test_ID, test_flux[0:2,:], test_ivar[0:2,:], lamost_label, 
            test_ID, test_flux, test_ivar)
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])
    ds.diagnostics_SNR(figname="%s_SNRdist.png" %date)

    ds.continuum_normalize_gaussian_smoothing(L=50)
    np.savez("output/%s_norm.npz" %date, ds.test_flux, ds.test_ivar)


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def test_step(date):
    wl = np.load("../run_2_train_on_good/wl.npz")['arr_0']
    test_ID = np.load("%s_test_ids.npz" %date)['arr_0']
    test_flux = np.load("%s_test_flux.npz" %date)['arr_0']
    test_ivar = np.load("%s_test_ivar.npz" %date)['arr_0']

    nlabels = 4
    nobj = len(test_ID)

    lamost_label_3 = np.load("%s_lamost_label.npz" %date)['arr_0']
    # add extra column to make it symmetric with the inferred test labels
    toadd = np.ones(nobj)[...,None]
    lamost_label = np.hstack((lamost_label_3, toadd))

    ds = dataset.Dataset(wl, test_ID, test_flux[0:2,:], test_ivar[0:2,:], lamost_label, 
            test_ID, test_flux, test_ivar)
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])

    m = model.CannonModel(2)
    m.coeffs = np.load("../run_5_train_on_good/coeffs.npz")['arr_0']
    m.scatters = np.load("../run_5_train_on_good/scatters.npz")['arr_0']
    m.chisqs = np.load("../run_5_train_on_good/chisqs.npz")['arr_0']
    m.pivots = np.load("../run_5_train_on_good/pivots.npz")['arr_0']

    nguesses = 4
    starting_guesses = np.zeros((nguesses,nlabels)) 
    hiT_hiG_hiM = np.array([  5.15273730e+03,   3.71762228e+00,   3.16861898e-01, 2.46907920e-02])
    hiT_hiG_loM = np.array([  5.16350098e+03,   3.45917511e+00,  -9.24426436e-01, 2.49296919e-01])
    loT_loG_hiM = np.array([  4.04936841e+03,   1.47109437e+00,   2.07210138e-01, 1.49733415e-02])
    loT_loG_loM = np.array([  4.00651318e+03,   8.35013509e-01,  -8.98257852e-01, 7.65705928e-02])
    starting_guesses[0,:] = hiT_hiG_hiM-m.pivots
    starting_guesses[1,:] = hiT_hiG_loM-m.pivots
    starting_guesses[2,:] = loT_loG_loM-m.pivots
    starting_guesses[3,:] = loT_loG_hiM-m.pivots

    labels = np.zeros((nguesses, nobj, nlabels)) # 4,10955,4
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

    np.savez("./%s_all_cannon_labels.npz" %date, best_labels)
    np.savez("./%s_cannon_label_chisq.npz" %date, best_chisq)
    np.savez("./%s_cannon_label_errs.npz" %date, best_errs)

    ds.test_label_vals = best_labels
    ds.diagnostics_survey_labels(figname="%s_survey_labels_triangle.png" %date)
    ds.diagnostics_1to1(figname = "%s_1to1_test_label.png" %date)


if __name__=="__main__":
    dir_dat = "/home/share/LAMOST/DR2"
    dates = os.listdir("%s/DR2_release" %dir_dat)
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])
    prep_data("20120201")
    prep_data("20121017")
    prep_data("20120105")
    prep_data("20140118")
    #for date in dates:
    #    print("running %s" %date)
    #    if glob.glob("output/%s_norm.npz" %date): print("already done")
    #    else:
    #        print("prepping %s" %date)
    #        prep_data(date)
            #bad = ['20111203', '20121208']
            #if date not in bad:  # don't know the reasons for this currently...
            #    prep_data(date)
            #    test_step(date)
