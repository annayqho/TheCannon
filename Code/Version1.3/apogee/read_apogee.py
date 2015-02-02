from __future__ import (absolute_import, division, print_function, unicode_literals)

"""Extract & continuum-normalize spectra from APOGEE .fits files."""

import numpy as np
import os

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

def get_spectra(dir_name):
    """
    Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

    Parameters
    ----------
    dirname: sequence
        a list of data file names of length nstars
    
    Returns
    -------
    lambdas: numpy ndarray of shape (npixels)
    norm_fluxes: numpy ndarray of shape (nstars, npixels)
    norm_ivars: numpy ndarray of shape (nstars, npixels)
    SNRs: numpy ndarray of shape (nstars)
    """
    
    files = [dir_name + "/" + filename for filename in os.listdir(dir_name)]
    files = list(sorted(files))
    nstars = len(files)

    LARGE = 1000000.
    for jj,fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        fluxes = np.array(file_in[1].data)
        if jj == 0: 
            npixels = len(fluxes)
            SNRs = np.zeros(nstars, dtype=float)
            norm_fluxes = np.zeros((nstars, npixels), dtype=float)
            norm_ivars = np.zeros(norm_fluxes.shape, dtype=float)
            #pixmasks = np.zeros(norm_fluxes.shape)
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl*(npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10**aval for aval in wl_full_log]
            lambdas = np.array(wl_full)
        flux_errs = np.array((file_in[2].data))
        badpix = get_pixmask(fluxes, flux_errs)
        fluxes = np.ma.array(fluxes, mask=badpix, fill_value=0.)
        flux_errs = np.ma.array(flux_errs, mask=badpix, fill_value=LARGE)
        SNRs[jj] = np.ma.median(fluxes/flux_errs)
        ones = np.ma.array(np.ones(npixels), mask=badpix)
        fluxes = np.ma.filled(fluxes)
        flux_errs = np.ma.filled(flux_errs)
        ivar = ones / (flux_errs**2)
        ivar = np.ma.filled(ivar, fill_value=0.)
        norm_flux, norm_ivar, continua = continuum_normalize_Chebyshev(lambdas, 
                                                                       fluxes, 
                                                                       flux_errs,                                                                        ivar)
        badpix2 = get_pixmask(norm_flux, 1./np.sqrt(norm_ivar))
        temp = np.ma.array(norm_flux, mask=badpix2, fill_value = 1.0)
        norm_fluxes[jj] = np.ma.filled(temp)
        temp = np.ma.array(norm_ivar, mask=badpix2, fill_value = 0.)
        norm_ivars[jj] = np.ma.filled(temp)
    
    print("Loaded {0:d} stellar spectra".format(len(files)))
    return lambdas, norm_fluxes, norm_ivars, SNRs

def get_pixmask(fluxes, flux_errs):
    """ Identify bad pixels in the spectrum

    Bad pixels are either when fluxes or errors are not finite or when reported
    errors are negative.

    Parameters
    ---------
    fluxes: ndarray
        flux array

    flux_errs: ndarray
        uncertainties on fluxes

    Returns
    -------
    mask: ndarray, dtype=bool
        array giving bad pixels as True values
    """

    bad_flux = ~np.isfinite(fluxes)
    bad_err = ~np.isfinite(flux_errs) or (flux_errs <= 0)
    bad_pix = bad_err or bad_flux

    return bad_pix

def continuum_normalize_Chebyshev(lambdas, fluxes, flux_errs, ivars):
    """"Continuum-normalizes the spectra.

    Fit a 2nd order Chebyshev polynomial to each segment 
    and divide each segment by its corresponding polynomial 

    Parameters
    ----------
    lambda: ndarray
        common wavelength
    fluxes: ndarray
        array of fluxes
    flux_errs: ndarray
        uncertainties on fluxes
    ivars: ndarray
        inverse variance matrix

    Returns
    -------
    norm_flux: ndarray
        continuum-normalized flux values (nstars, npixels)
    norm_ivar: ndarray
        normalized inverse variance (nstars, npixels)
    """
    continua = np.zeros(lambdas.shape)
    norm_flux = np.zeros(fluxes.shape)
    norm_flux_err = np.zeros(flux_errs.shape)
    norm_ivar = np.zeros(ivars.shape)
    
    # list of "true" continuum pix, det. here by the Cannon
    pixlist = list(np.loadtxt("pixtest4.txt", dtype=int, usecols=(0,), unpack=1))
    contmask = np.ones(len(lambdas), dtype=bool)
    contmask[pixlist] = 0
    ivars[contmask] = 0. # ignore non-cont pixels 
    
    # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels
    ranges = [[371,3192], [3697,5997], [6461,8255]]
    for i in range(len(ranges)):
        start, stop = ranges[i][0], ranges[i][1]
        flux = fluxes[start:stop]
        flux_err = flux_errs[start:stop]
        lambda_cut = lambdas[start:stop]
        ivar = ivars[start:stop]
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=lambda_cut, y=flux, 
                                                    w=ivar, deg=3)
        continua[start:stop] = fit(lambda_cut)
        norm_flux[start:stop] = flux/fit(lambda_cut)
        norm_flux_err[start:stop] = flux_err/fit(lambda_cut)
        norm_ivar[start:stop] = 1. / norm_flux_err[start:stop]**2
    return norm_flux, norm_ivar, continua
