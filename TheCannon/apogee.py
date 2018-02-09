""" Functions for reading in APOGEE spectra and training labels """

from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
import os
import sys
import matplotlib.pyplot as plt
from astropy.io import ascii
from TheCannon import *

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

def get_pixmask(fluxes, flux_errs):
    """ Create and return a bad pixel mask for an APOGEE spectrum

    Bad pixels are defined as follows: fluxes or errors are not finite, or 
    reported errors are <= 0, or fluxes are 0

    Parameters
    ----------
    fluxes: ndarray
        Flux data values 

    flux_errs: ndarray
        Flux uncertainty data values 

    Returns
    -------
    mask: ndarray
        Bad pixel mask, value of True corresponds to bad pixels
    """
    bad_flux = np.logical_or(~np.isfinite(fluxes), fluxes == 0)
    bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
    bad_pix = bad_err | bad_flux
    return bad_pix


def get_starmask(ids, labels, aspcapflag, paramflag):
    """ Identifies which APOGEE objects have unreliable physical parameters,
    as laid out in Holzman et al 2015 and on the APOGEE DR12 website

    Parameters
    ----------
    data: np array
        all APOGEE DR12 IDs and labels

    Returns
    -------
    bad: np array
        mask where 1 corresponds to a star with unreliable parameters
    """
    # teff outside range (4000,6000) K and logg < 0
    teff = labels[0,:]
    bad_teff = np.logical_or(teff < 4000, teff > 6000)
    logg = labels[1,:]
    bad_logg = logg < 0
    cuts = bad_teff | bad_logg

    # STAR_WARN flag set (TEFF, LOGG, CHI2, COLORTE, ROTATION, SN)
    # M_H_WARN, ALPHAFE_WARN not included in the above, so do them separately
    star_warn = np.bitwise_and(aspcapflag, 2**7) != 0
    star_bad = np.bitwise_and(aspcapflag, 2**23) != 0
    feh_warn = np.bitwise_and(aspcapflag, 2**3) != 0
    alpha_warn = np.bitwise_and(aspcapflag, 2**4) != 0
    aspcapflag_bad = star_warn | star_bad | feh_warn | alpha_warn

    # separate element flags
    teff_flag = paramflag[:,0] != 0
    logg_flag = paramflag[:,1] != 0
    feh_flag = paramflag[:,3] != 0
    alpha_flag = paramflag[:,4] != 0
    paramflag_bad = teff_flag | logg_flag | feh_flag | alpha_flag

    return cuts | aspcapflag_bad | paramflag_bad 


def load_spectra(data_dir):
    """ Reads wavelength, flux, and flux uncertainty data from apogee fits files

    Parameters
    ----------
    data_dir: str
        Name of the directory containing all of the data files

    Returns
    -------
    wl: ndarray
        Rest-frame wavelength vector

    fluxes: ndarray
        Flux data values

    ivars: ndarray
        Inverse variance values corresponding to flux values
    """
    print("Loading spectra from directory %s" %data_dir)
    files = list(sorted([data_dir + "/" + filename
             for filename in os.listdir(data_dir) if filename.endswith('fits')]))
    nstars = len(files)  
    for jj, fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        flux = np.array(file_in[1].data)
        if jj == 0:
            npixels = len(flux)
            fluxes = np.zeros((nstars, npixels), dtype=float)
            ivars = np.zeros(fluxes.shape, dtype=float)
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl * (npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10 ** aval for aval in wl_full_log]
            wl = np.array(wl_full)
        flux_err = np.array((file_in[2].data))
        badpix = get_pixmask(flux, flux_err)
        ivar = np.zeros(npixels)
        ivar[~badpix] = 1. / flux_err[~badpix]**2
        fluxes[jj,:] = flux
        ivars[jj,:] = ivar
    # convert filenames to actual IDs
    names = np.array([f.split('-')[2][:-5] for f in files])
    print("Spectra loaded")
    # make sure they are numpy arrays
    return np.array(names), np.array(wl), np.array(fluxes), np.array(ivars)


def load_labels(filename):
    """ Extracts reference labels from a file

    Parameters
    ----------
    filename: str
        Name of the file containing the table of reference labels

    ids: array
        The IDs of stars to retrieve labels for

    Returns
    -------
    labels: ndarray
        Reference label values for all reference objects
    """
    print("Loading reference labels from file %s" %filename)
    data = ascii.read(filename)
    ids = data['ID']
    inds = ids.argsort()
    ids = ids[inds]
    teff = data['Teff_{corr}']
    teff = teff[inds]
    logg = data['logg_{corr}']
    logg = logg[inds]
    mh = data['[M/H]_{corr}']
    mh = mh[inds]
    return np.vstack((teff,logg,mh)).T 


def continuum_normalize_training(ds):
    """ 
    Continuum normalize the training set, using an iterative Cannon approach

    Parameters
    ----------
    ds: dataset object

    Returns
    -------
    updated dataset object
    
    """

    # To initialize the continuum-pixel determination, we define a
    # preliminary pseudo-continuum-normalization by using a polynomial
    # fit to an upper quantile (in this case 90%) of the spectra, determined
    # from a running median
    # this is SNR-dependent
    pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(
            q=0.90, delta_lambda=50)
    ds.tr_flux = pseudo_tr_flux
    ds.tr_ivar = pseudo_tr_ivar

    # Run the training step
    m = model.CannonModel(2)
    m.fit(ds)

    # Baseline spectrum
    baseline_spec = m.coeffs[:,0]

    # Flux cut: 1 +/- 0.15 (0.985 - 1.015)
    fcut = (np.abs(baseline_spec - 1) <= 0.15)

    # Smallest percentiles of the first order coefficients
    ccut_1 = np.logical_and(
            np.abs(m.coeffs[:,1]) < 1.0e-5, np.abs(m.coeffs[:,1] > 0))
    ccut_2 = np.logical_and(
            np.abs(m.coeffs[:,2]) < 0.0045, np.abs(m.coeffs[:,2] > 0))
    ccut_3 = np.logical_and(
            np.abs(m.coeffs[:,3]) < 0.0085, np.abs(m.coeffs[:,3] > 0))
    ccut12 = np.logical_and(ccut_1, ccut_2)
    ccut = np.logical_and(ccut12, ccut_3)

    # Choose the continuum pixels
    contpix = np.logical_and(fcut, ccut)
    ds.set_continuum(contpix)

    # Fit a sinusoid to these pixels, using the inverse variance weighting
    # Adding an additional error term that is set to 0 for continuum pixels
    # and a large error value for all other pixels so that the new error 
    # term becomes 
    #err2 = err1 + err[0 OR LARGE]
    cont = dataset.fit_continuum(3, "sinusoid")
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
            dataset.continuum_normalize(cont)
   
    ds.tr_flux = norm_tr_flux
    ds.tr_ivar = norm_tr_ivar
    ds.test_flux = norm_test_flux
    ds.test_ivar = norm_test_ivar

    return ds

if  __name__ =='__main__':
    make_apogee_label_file()
