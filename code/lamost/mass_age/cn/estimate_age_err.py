""" Estimate the uncertainty in the age measurement.

Sample from [Fe/H], [C/M], [N/M], Teff, logg to estimate
the width of the age measurement.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/annaho/Dropbox/Research/TheCannon/code/lamost/mass_age")
from mass_age_functions import *

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/xval_with_cuts"
lab = np.load(DATA_DIR + "/xval_cannon_label_vals.npz")['arr_0']
ref = np.load(DATA_DIR + "/ref_label.npz")['arr_0']
snr = np.load(DATA_DIR + "/ref_snr.npz")['arr_0']
snr_binned = np.zeros(snr.shape)
nobj, nlab = ref.shape

#snr_bins = np.array([10,30,50,70,90,110])
snr_bins = np.arange(5,110,10)
nbins = len(snr_bins)
scatters = np.zeros((nbins,nlab))
errs = np.zeros(scatters.shape)

for ii,snr_bin in enumerate(snr_bins):
    choose = np.abs(snr-snr_bin)<=5
    snr_binned[choose] = snr_bin
    nobj = sum(choose)
    diff = lab[choose] - ref[choose]
    #scatters[ii,:] = np.std(diff[choose], axis=0)
    # bootstrap 100 times
    nbs = 100
    samples = np.random.randint(0,nobj,(nbs,nobj)) # (nbs, nobj)
    stdev = np.std(diff[samples], axis=1)
    scatters[ii,:] = np.mean(stdev, axis=0)
    errs[ii,:] = np.std(stdev, axis=0)

# plt.scatter(snr_bins, scatters[:,i])
# plt.errorbar(snr_bins, scatters[:,i], yerr=errs[:,i])
nsamples = 10000

def get_samples(obj,i):
    mean_val = lab[obj,i]
    sig_val = np.zeros(mean_val.shape)
    choose = snr_bins == snr_binned[obj]
    sig_val = scatters[choose,i]
    dist = np.random.normal(loc=mean_val, scale=sig_val, size=nsamples)
    return dist

choose_obj = 201
teff = get_samples(choose_obj,0)
logg = get_samples(choose_obj,1)
feh = get_samples(choose_obj,2)
cm = get_samples(choose_obj,3)
nm = get_samples(choose_obj,4)

ages = 10**calc_logAge(feh, cm, nm, teff, logg)
#for sample in nsamples:
#    calc_logAge(feh, cm, nm, teff, logg)
