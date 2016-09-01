""" Estimate age and its uncertainty.

Sample from [Fe/H], [C/M], [N/M], Teff, logg to estimate
the width of the age measurement.
"""

import numpy as np
from scipy.stats.mstats import mode
import matplotlib.pyplot as plt
import sys
sys.path.append(
"/Users/annaho/Dropbox/Research/TheCannon/code/lamost/mass_age")
from mass_age_functions import *
from marie_cuts import get_mask

def get_samples(lab, obj,i, snr_bins, snr_binned, scatters, nsamples):
    mean_val = lab[obj,i]
    sig_val = np.zeros(mean_val.shape)
    # figure out which bin its SNR corresponds to
    choose = snr_bins == snr_binned[obj]
    sig_val = scatters[choose,i]
    dist = np.random.normal(loc=mean_val, scale=sig_val, size=nsamples)
    return dist


def estimate_age(ref_label, cannon_ref_label, ref_snr, test_label, test_snr):
    # Use the ref labels to create SNR - scatter mapping
    print("Using ref labels to create SNR scatter mapping")
    ref_snr[ref_snr > 300] = 300
    snr_binned = np.zeros(ref_snr.shape)
    nobj, nlab = ref_label.shape

    snr_bins = np.arange(5,300,10)
    nbins = len(snr_bins)
    scatters = np.zeros((nbins,nlab))
    errs = np.zeros(scatters.shape)

    for ii,snr_bin in enumerate(snr_bins):
        choose = np.abs(ref_snr-snr_bin)<=5
        snr_binned[choose] = snr_bin
        nobj_in_bin = sum(choose)
        diff = cannon_ref_label[choose] - ref_label[choose]
        #scatters[ii,:] = np.std(diff[choose], axis=0)
        # bootstrap 100 times
        nbs = 100
        samples = np.random.randint(
                0,nobj_in_bin,(nbs,nobj_in_bin)) # (nbs, nobj)
        stdev = np.std(diff[samples], axis=1)
        scatters[ii,:] = np.mean(stdev, axis=0)
        errs[ii,:] = np.std(stdev, axis=0)

    # Now, for all of the test labels, estimate ages & age errors
    nobj, nlab = test_label.shape
    print("Estimating ages & age errors")
    ages = np.zeros(nobj)
    masses = np.zeros(nobj)
    age_errs = np.zeros(ages.shape)
    mass_errs = np.zeros(masses.shape)
    nsamples = 1000

    test_snr[test_snr > 300] = 300
    snr_binned = np.zeros(test_snr.shape)
    for ii,snr_bin in enumerate(snr_bins):
        choose = np.abs(test_snr-snr_bin)<=5
        snr_binned[choose] = snr_bin

    for choose_obj in np.arange(nobj):
        teff = get_samples(
                test_label, choose_obj, 0,
                snr_bins, snr_binned, scatters, nsamples)
        logg = get_samples(
                test_label, choose_obj, 1, 
                snr_bins, snr_binned, scatters, nsamples)
        feh = get_samples(
                test_label, choose_obj, 2, 
                snr_bins, snr_binned, scatters, nsamples)
        cm = get_samples(
                test_label, choose_obj, 3, 
                snr_bins, snr_binned, scatters, nsamples)
        nm = get_samples(
                test_label, choose_obj, 4, 
                snr_bins, snr_binned, scatters, nsamples)

        # return the mode and the 68th percentile
        age_samples = calc_logAge(feh, cm, nm, teff, logg)
        mass_samples = np.log10(calc_mass_2(feh, cm, nm, teff, logg)) # in log(Mass)
        #plt.hist(age_samples, bins=20, range=(0,2))
        #plt.show()
        ages[choose_obj] = np.median(age_samples)
        masses[choose_obj] = np.median(mass_samples)
        percentile = 0.68
        dist = (1-percentile)/2
        bottom = np.sort(age_samples)[int(dist*nsamples)]
        top = np.sort(age_samples)[int((1-dist)*nsamples)]
        age_errs[choose_obj] = (top-bottom)/2
        bottom = np.sort(mass_samples)[int(dist*nsamples)]
        top = np.sort(mass_samples)[int((1-dist)*nsamples)]
        mass_errs[choose_obj] = (top-bottom)/2

    return ages, age_errs, masses, mass_errs


if __name__=="__main__":
    DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/xval_with_cuts"
    lab = np.load(DATA_DIR + "/xval_cannon_label_vals.npz")['arr_0']
    ref = np.load(DATA_DIR + "/ref_label.npz")['arr_0']
    snr = np.load(DATA_DIR + "/ref_snr.npz")['arr_0']
    age, age_err = estimate_age(ref, lab, snr, lab, snr)
