""" Mass production for the xcalib paper. 
All you need to change each time is the date you want to run. """

import numpy as np
import pickle
import glob
import os
from matplotlib import rc
sys.path.insert(0, '/home/annaho/xcalib/TheCannon')
from lamost import load_spectra
from TheCannon import dataset
from TheCannon import model
from lamost import load_spectra
from astropy.table import Table
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
rc('text', usetex=True)
rc('font', family='serif')


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def run(date):
    # Training step has already been completed. Load the model,
    spectral_model = model.CannonModel(2) # 2 = quadratic model
    spectral_model.coeffs = np.load("./coeffs.npz")['arr_0']
    spectral_model.scatters = np.load("./scatter.npz")['arr_0']
    spectral_model.chisqs = np.load("./chisqs.npz")['arr_0']
    spectral_model.pivots = np.load("./pivots.npz")['arr_0']

    # Load the wavelength array
    wl = np.load("wl.npz")['arr_0']

    # Load the test set,
    test_ID = np.loadtxt("test_obj/%s_test_obj.txt" %date, dtype=str)
    print("%s test objects" %len(test_ID))
    dir_dat = "/home/share/LAMOST/DR2/DR2_release"
    test_IDs, wl, test_flux, test_ivar = load_spectra(dir_dat, test_ID)
    np.savez("output/%s_ids" %date, test_IDs)
    #np.savez("./%s_data_raw" %date, test_flux, test_ivar)

    # Load the corresponding LAMOST labels,
    labels = np.load("lamost_labels/lamost_labels_%s.npz" %date)['arr_0']
    inds = np.array([np.where(labels[:,0]==a)[0][0] for a in test_IDs]) 
    nstars = len(test_IDs)
    lamost_labels = np.zeros((nstars,4))
    lamost_labels[:,0:3] = labels[inds,:][:,1:].astype(float) 
    np.savez("output/%s_lamost_label" %date, lamost_labels)
    
    # Set dataset object
    data = dataset.Dataset(
            wl, test_IDs, test_flux, test_ivar, 
            lamost_labels, test_IDs, test_flux, test_ivar)

    # set the headers for plotting
    data.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])
    
    # Plot SNR distribution
    data.diagnostics_SNR(figname="%s_SNRdist.png" %date)
    np.savez("output/%s_SNR" %date, data.test_SNR)

    # Continuum normalize, 
    filename = "output/%s_norm.npz" %date
    if glob.glob(filename):
        print("already cont normalized")
        data.test_flux = np.load(filename)['arr_0']
        data.test_ivar = np.load(filename)['arr_1']
    else:
        data.tr_ID = data.tr_ID[0]
        data.tr_flux = data.tr_flux[0,:]
        data.tr_ivar = data.tr_ivar[0,:]
        data.continuum_normalize_gaussian_smoothing(L=50)
        np.savez("output/%s_norm" %date, data.test_flux, data.test_ivar)

    # Infer labels 
    errs, chisq = spectral_model.infer_labels(data)
    np.savez("output/%s_cannon_labels.npz" %date, data.test_label_vals)
    np.savez("./%s_formal_errors.npz" %date, errs)
    np.savez("./%s_chisq.npz" %date, chisq)

    # Make plots
    data.test_label_vals = data.test_label_vals[:,0:3] # so it doesn't try alpha
    data.set_label_names(['T_{eff}', '\log g', '[M/H]'])
    data.diagnostics_1to1(figname="%s_1to1_label" %date)

    # Find chi sq of fit
    # lvec = _get_lvec(list(data.test_label_vals[jj,:]-model.pivots))
    # chi = data.tr_flux[jj,:] - (np.dot(coeffs, lvec) + model.coeffs[:,0])
    # chisq_star = sum(chi**2)
    # dof = npix - nparam

if __name__ == "__main__":
    #date = '20130324'
    #print(date)
    #run(date)
    
#    dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
#    dates = np.array(dates)
#    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
#    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
#    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])
#    dates = np.delete(dates, np.where(dates=='20140330')[0][0]) # no obj
#    dates = np.delete(dates, np.where(dates=='20121028')[0][0]) # no obj
#    for date in dates[366:]: 
#        print(date)
#	run(date)
        # run(date)
	# DR1 = dates before sept 2013
        # if np.logical_and(int(date[0:4]) < 2014, int(date[4:6]) < 9): 
        #     print("is DR1")
        #     if len(glob.glob("%s_1to1_label_0.png" %date)) == 0:
        #         print("isn't done yet, running TC")
        #         run(date)

    a = glob.glob("output/2*cannon_labels.npz")
    b = glob.glob("output/2*ids.npz")
    c = glob.glob("output/2*SNR.npz")
    d = glob.glob("output/2*formal_errors.npz")
    e = glob.glob("output/2*chisq.npz")
    ids_all = np.array([])
    teff_all = np.array([])
    logg_all = np.array([])
    feh_all = np.array([])
    alpha_all = np.array([])
    SNRs_all = np.array([])
    errs_all = np.array([])
    chisq_all = np.array([])
    print("cannon labels")
    for filename in a:
        labels = np.load(filename)['arr_0']
        teff_all = np.append(teff_all, labels[:,0])
        logg_all = np.append(logg_all, labels[:,1])
        feh_all = np.append(feh_all, labels[:,2])
        alpha_all = np.append(alpha_all, labels[:,3])
    print("IDs")
    for filename in b:
        ids = np.load(filename)['arr_0']
        ids_all = np.append(ids_all, ids)
    print("SNRs")
    for filename in c:
        SNR = np.load(filename)['arr_0']
        SNRs_all = np.append(SNRs_all, SNR) 
    print("Formal Errors")
    for filename in d:
        errs = np.load(filename)['arr_0']
        errs_all = np.append(errs_all, errs)
    print("Chi Sq")
    for filename in e:
        chisq = np.load(filename)['arr_0']
        chisq_all = np.append(chisq_all, chisq)
    np.savez("DR2/teff_all.npz",teff_all)
    np.savez("DR2/logg_all.npz",logg_all)
    np.savez("DR2/feh_all.npz",feh_all)
    np.savez("DR2/alpha_all.npz", alpha_all)
    np.savez("DR2/ids_all.npz", ids_all)
    np.savez("DR2/chisq_all.npz", chisq_all)
    np.savez("DR2/SNRs_all.npz", SNRs_all)
    np.savez("DR2/errs_all.npz", errs_all)
