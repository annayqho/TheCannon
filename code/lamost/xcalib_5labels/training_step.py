"""
Train TC on all *good* objects in the 11,057 overlap set 
(9594 objects)
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import os
import pyfits
from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
from TheCannon import lamost
from TheCannon import dataset
from TheCannon import model


def load_data():
    print("Loading all data...")
    a = pyfits.open("../data/label_file.fits") 
    tbdata = a[1].data
    a.close()
    apogee_teff = tbdata['apogee_teff']
    apogee_logg = tbdata['apogee_logg']
    apogee_mh = tbdata['apogee_mh']
    apogee_alpham = tbdata['apogee_alpham']
    apogee_reddening = tbdata['AK_WISE']
    tr_label = np.vstack((apogee_teff,apogee_logg,apogee_mh,apogee_alpham,apogee_reddening)).T
    tr_id_full = tbdata['lamost_id']
    tr_id = np.array([val.strip() for val in tr_id_full])

    # Load data for all 11,057 overlap objects & select training data
    all_id = np.load("../data/all_ids.npz")['arr_0']
    all_flux = np.load("../data/test_flux.npz")['arr_0']
    all_ivar = np.load("../data/test_ivar.npz")['arr_0']

    print("Selecting training data...")
    good = np.array([np.where(all_id==f)[0][0] for f in tr_id])

    good_flux = all_flux[good,:] 
    good_ivar = all_ivar[good,:]

    np.savez("tr_id.npz", tr_id)
    np.savez("tr_label.npz", tr_label)
    np.savez("tr_flux.npz", good_flux)
    np.savez("tr_ivar.npz", good_ivar)


def train():
    # Load training set
    wl = np.load("../data/wl.npz")['arr_0']
    tr_id = np.load("tr_id.npz")['arr_0']
    tr_label = np.load("tr_label.npz")['arr_0']
    tr_flux = np.load("tr_flux.npz")['arr_0']
    tr_ivar = np.load("tr_ivar.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label, tr_id, tr_flux, tr_ivar)
    ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]', 'AKWISE'])
    ds.diagnostics_SNR()
    ds.diagnostics_ref_labels()
    np.savez("./tr_snr.npz", ds.tr_SNR)

    m = model.CannonModel(2)
    m.fit(ds)
    np.savez("./coeffs.npz", m.coeffs)
    np.savez("./scatters.npz", m.scatters)
    np.savez("./chisqs.npz", m.chisqs)
    np.savez("./pivots.npz", m.pivots)
    m.diagnostics_leading_coeffs(ds)
    m.diagnostics_leading_coeffs_triangle(ds)
    m.diagnostics_plot_chisq(ds)


if __name__=="__main__":
    #load_data()
    train()
