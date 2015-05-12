from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
import os
import sys
from .helpers import Table
import matplotlib.pyplot as plt

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

def get_pixmask(fluxes, flux_errs):
    """ Return a mask array of bad pixels for one object's spectrum

    Bad pixels are defined as follows: fluxes or errors are not finite, or 
    reported errors are negative, or the standard deviation of the fluxes
    across all the stars is zero (if that pixel is exactly the same, then
    we're looking at the gaps in the spectrum.)

    Parameters
    ----------
    fluxes: ndarray
        flux array

    flux_errs: ndarray
        measurement uncertainties on fluxes

    Returns
    -------
    mask: ndarray, dtype=bool
        array giving bad pixels as True values
    """
    bad_flux = (~np.isfinite(fluxes)) 
    bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
    bad_pix = bad_err | bad_flux

    return bad_pix

def load_spectra(data_dir):
    """
    Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

    Returns
    -------
    IDs: list of length nstars
        stellar IDs
    
    wl: numpy ndarray of length npixels
        rest-frame wavelength vector

    fluxes: numpy ndarray of shape (nstars, npixels)
        training set or test set pixel intensities

    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes
        
    SNRs: numpy ndarray of length nstars
    """
    print("Loading spectra from directory %s" %data_dir)
    files = list(sorted([data_dir + "/" + filename
             for filename in os.listdir(data_dir)]))
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
        flux = np.ma.array(flux, mask=badpix)
        flux_err = np.ma.array(flux_err, mask=badpix)
        ones = np.ma.array(np.ones(npixels), mask=badpix)
        ivar = ones / flux_err**2
        ivar = np.ma.filled(ivar, fill_value=0.)
        fluxes[jj,:] = flux
        ivars[jj,:] = ivar

    print("Spectra loaded")
    return wl, fluxes, ivars

def load_labels(filename):
    print("Loading reference labels from file %s" %filename)
    data = Table(filename)
    data.sort('id')
    label_names = data.keys()[1:] # ignore id
    nlabels = len(label_names)
    print('%s labels:' %nlabels)
    print(label_names)
    labels = np.array([data[k] for k in label_names], dtype=float).T
    return labels 
