# This is a demonstration of how to run The Cannon on RAVE Data

import numpy as np
import pickle
import glob
from scipy.io.idl import readsav

# STEP 1: PREPARE DATA

# The Cannon needs: length-L wavelength vec 
# an NxL block of training set pixel fluxes and corresponding ivars
# an NxK block of training label values
# an MxL block of test set pixel vals and corresponding ivars

# training set
inputf = readsav('RAVE_DR4_calibration_data.save') 
items = inputf.items()
data = items[0][1]

tr_flux = data['spectrum'][0].T # shape (807, 839) = (nstars, npix)
npix = tr_flux.shape[1]
nstars = tr_flux.shape[0]

teff = data['teff'][0] # length 807
logg = data['logg'][0] # length 807
feh = data['feh'][0]
tr_label = np.vstack((teff, logg, feh)).T

snr = np.zeros(nstars)
snr.fill(100) # a guess for what the SNR could be
tr_ivar = (snr[:,None]/tr_flux)**2

# test set
def read_test(filename): 
    inputf = readsav(filename)
    items = inputf.items()
    data = items[0][1]
    sp = data['obs_sp'] # (75437, 839) 
    test_flux = np.zeros((len(sp), len(sp[0])))
    for jj in range(0, len(sp)):
        test_flux[jj,:] = sp[jj]
    snr = data['snr']
    test_ivar = (snr[:,None]/test_flux)**2
    wl = data['lambda'][0] # assuming they're all the same... 
    npix = len(wl)
    nstars = len(snr)
    return (test_flux, test_ivar, wl)

if glob.glob('rave_test_data.p'):
    (wl, test_flux, test_ivar) = pickle.load(
            open('rave_test_data.p', 'r'))

else:
    a = read_test('2009K1_parameters.save')
    b = read_test('2013K1_parameters.save')

    test_flux = np.vstack((a[0], b[0]))
    test_ivar = np.vstack((a[1], b[1]))
    wl = a[2] # same for both sets
    pickle.dump((wl, test_flux, test_ivar), open("rave_test_data.p", "w"), -1)

# initialize a dataset object
dataset = Dataset(wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar)

# STEP 2: diagnostic plots for input spectra and reference labels
dataset.diagnostics_SNR()
dataset.diagnostics_ref_labels()


