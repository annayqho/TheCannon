""" Mass production for the xcalib paper. 
All you need to change each time is the date you want to run. """

import numpy as np
import pickle
import glob
import os
from matplotlib import rc
from lamost import load_spectra, load_labels
from TheCannon import dataset
from TheCannon import model

rc('text', usetex=True)
rc('font', family='serif')

def run(date):
    # Training step has already been completed. Load the model,
    spectral_model = model.CannonModel(2) # 2 = quadratic model
    spectral_model.coeffs = np.load("./model_coeffs.npz")['arr_0']
    spectral_model.scatters = np.load("./model_scatter.npz")['arr_0']
    spectral_model.chisqs = np.load("./model_chisqs.npz")['arr_0']
    spectral_model.pivots = np.load("./model_pivots.npz")['arr_0']

    # Load the wavelength array
    wl = np.load("wl.npz")['arr_0']

    # Load the test set,
    test_ID = np.loadtxt("lamost_dr2/%s_test_obj.txt" %date, dtype=str)
    print("%s test objects" %len(test_ID))
    dir_dat = "lamost_dr2/DR2_release"
    test_IDs, wl, test_flux, test_ivar = load_spectra(dir_dat, test_ID)
    np.savez("./%s_ids" %date, test_IDs)
    #np.savez("./%s_data_raw" %date, test_flux, test_ivar)

    # Load the corresponding LAMOST labels,
    labels = np.load("lamost_dr2/lamost_labels_%s.npz" %date)['arr_0']
    inds = np.array([np.where(labels[:,0]==a)[0][0] for a in test_IDs]) 
    nstars = len(test_IDs)
    lamost_labels = np.zeros((nstars,4))
    lamost_labels[:,0:3] = labels[inds,:][:,1:].astype(float) 
    np.savez("./%s_tr_label" %date, lamost_labels)
    
    # Set dataset object
    data = dataset.Dataset(
            wl, test_IDs, test_flux, test_ivar, 
            lamost_labels, test_IDs, test_flux, test_ivar)

    # set the headers for plotting
    data.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])
    
    # Plot SNR distribution
    data.diagnostics_SNR(figname="%s_SNRdist.png" %date)

    # Continuum normalize,
    data.tr_ID = data.tr_ID[0]
    data.tr_flux = data.tr_flux[0,:]
    data.tr_ivar = data.tr_ivar[0,:]
    data.continuum_normalize_gaussian_smoothing(L=50)
    np.savez("./%s_norm" %date, data.test_flux, data.test_ivar)

    # Infer labels 
    spectral_model.infer_labels(data)
    np.savez("./%s_cannon_labels.npz" %date, data.test_label_vals)

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
    
    dates = os.listdir("lamost_dr2/DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    for date in dates: 
        print(date)
        # DR1 = dates before sept 2013
        if np.logical_and(int(date[0:4]) < 2014, int(date[4:6]) < 9): 
            print("is DR1")
            if len(glob.glob("%s_1to1_label_0.png" %date)) == 0:
                print("isn't done yet, running TC")
                run(date)

    a = glob.glob("./*cannon_labels.npz")
    teff_all = np.array([])
    logg_all = np.array([])
    feh_all = np.array([])
    alpha_all = np.array([])
    for filename in a:
        labels = np.load(filename)['arr_0']
        teff_all = np.append(teff_all, labels[:,0])
        logg_all = np.append(logg_all, labels[:,1])
        feh_all = np.append(feh_all, labels[:,2])
        alpha_all = np.append(alpha_all, labels[:,3])
