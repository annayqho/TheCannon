""" Estimate age and its uncertainty.

Sample from [Fe/H], [C/M], [N/M], Teff, logg to estimate
the width of the age measurement.
"""

import numpy as np
from scipy.stats.mstats import mode
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/annaho/Dropbox/Research/TheCannon/code/lamost/mass_age")
from mass_age_functions import *
from marie_cuts import get_mask

def get_samples(lab, obj,i, snr_bins, snr_binned, scatters, nsamples):
    mean_val = lab[obj,i]
    sig_val = np.zeros(mean_val.shape)
    choose = snr_bins == snr_binned[obj]
    sig_val = scatters[choose,i]
    dist = np.random.normal(loc=mean_val, scale=sig_val, size=nsamples)
    return dist


def estimate_age():
    DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/xval_with_cuts"
    lab = np.load(DATA_DIR + "/xval_cannon_label_vals.npz")['arr_0']
    ref = np.load(DATA_DIR + "/ref_label.npz")['arr_0']
    snr = np.load(DATA_DIR + "/ref_snr.npz")['arr_0']
    snr[snr > 300] = 300
    snr_binned = np.zeros(snr.shape)
    nobj, nlab = ref.shape
    ages = np.zeros(nobj)
    age_errs = np.zeros(ages.shape)

    #snr_bins = np.array([10,30,50,70,90,110])
    snr_bins = np.arange(5,300,10)
    nbins = len(snr_bins)
    scatters = np.zeros((nbins,nlab))
    errs = np.zeros(scatters.shape)

    for ii,snr_bin in enumerate(snr_bins):
        choose = np.abs(snr-snr_bin)<=5
        snr_binned[choose] = snr_bin
        nobj_in_bin = sum(choose)
        diff = lab[choose] - ref[choose]
        #scatters[ii,:] = np.std(diff[choose], axis=0)
        # bootstrap 100 times
        nbs = 100
        samples = np.random.randint(0,nobj_in_bin,(nbs,nobj_in_bin)) # (nbs, nobj)
        stdev = np.std(diff[samples], axis=1)
        scatters[ii,:] = np.mean(stdev, axis=0)
        errs[ii,:] = np.std(stdev, axis=0)

    # plt.scatter(snr_bins, scatters[:,i])
    # plt.errorbar(snr_bins, scatters[:,i], yerr=errs[:,i])
    nsamples = 1000

    for choose_obj in np.arange(nobj):
    #for choose_obj in np.arange(0, 1):
        teff = get_samples(
                lab, choose_obj, 0, snr_bins, snr_binned, scatters, nsamples)
        logg = get_samples(
                lab, choose_obj, 1, snr_bins, snr_binned, scatters, nsamples)
        feh = get_samples(
                lab, choose_obj, 2, snr_bins, snr_binned, scatters, nsamples)
        cm = get_samples(
                lab, choose_obj, 3, snr_bins, snr_binned, scatters, nsamples)
        nm = get_samples(
                lab, choose_obj, 4, snr_bins, snr_binned, scatters, nsamples)

        # return the mode and the 68th percentile
        age_samples = calc_logAge(feh, cm, nm, teff, logg)
        #plt.hist(age_samples, bins=20, range=(0,2))
        #plt.show()
        ages[choose_obj] = np.median(age_samples)
        percentile = 0.68
        dist = (1-percentile)/2
        bottom = np.sort(age_samples)[int(dist*nsamples)]
        top = np.sort(age_samples)[int((1-dist)*nsamples)]
        age_errs[choose_obj] = (top-bottom)/2
    return ages, age_errs


if __name__=="__main__":
    ages, age_errs = estimate_age()
