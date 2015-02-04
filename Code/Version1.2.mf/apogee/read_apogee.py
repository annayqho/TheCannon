from __future__ import (absolute_import, division, print_function, unicode_literals)

"""Extract & continuum-normalize spectra from APOGEE .fits files."""

import numpy as np
import os

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

# imported but not used
# import matplotlib.pyplot as plt


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
        common wavelength of the spectra

    spectra: 3D numpy ndarray of shape (nstars, npixels, 2)
        array of spectra expected with the following format:
            spectra[:,:,0] = flux values
            spectra[:,:,1] = flux err values
    """
    files = [dir_name + "/" + filename for filename in os.listdir(dir_name)]
    files = list(sorted(files))  # replaced by builtin

    # FIXME: hard coded value
    LARGE = 1000000.
    nstars = len(files)   # moved outside the loop
    for jj, fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        fluxes = np.array(file_in[1].data)
        if jj == 0:
            npixels = len(fluxes)
            SNRs = np.zeros(nstars, dtype=float)   # imposing the type is faster
            norm_fluxes = np.zeros((nstars, npixels), dtype=float)
            norm_ivars = np.zeros(norm_fluxes.shape, dtype=float)
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl * (npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10 ** aval for aval in wl_full_log]
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
                                                                       flux_errs,
                                                                       ivar)
        badpix2 = get_pixmask(norm_flux, 1. / np.sqrt(norm_ivar))
        temp = np.ma.array(norm_flux, mask=badpix2, fill_value=1.0)
        norm_fluxes[jj] = np.ma.filled(temp)
        temp = np.ma.array(norm_ivar, mask=badpix2, fill_value=0.)
        norm_ivars[jj] = np.ma.filled(temp)

    # len gives an int and you print a string
    # print("Loaded %s stellar spectra" %len(files))
    # new formatting is more flexible
    print("Loaded {0:d} stellar spectra".format(len(files)))
    return lambdas, norm_fluxes, norm_ivars, SNRs


def get_pixmask(fluxes, flux_errs):
    """ return a mask array of bad pixels

    bad pixels are either when fluxes or errors are not finite or when reported
    errors are negative.

    Parameters
    ----------

    fluxes: ndarray
        flux array

    flux_errs: ndarray
        uncertainties on fluxes

    returns
    -------
    mask: ndarray, dtype=bool
        array giving bad pixels as True values
    """
    # bad_flux = np.isinf(fluxes)
    # bad_err = np.isinf(flux_errs) or (flux_errs <= 0)
    # bad_err = np.logical_or(np.isinf(flux_errs), flux_errs <= 0)
    bad_flux = ~np.isfinite(fluxes)   # you probably want to clean nans as well
    bad_err = ~np.isfinite(flux_errs) or (flux_errs <= 0)
    bad_pix = bad_err or bad_flux

    return bad_pix


# Plotting function?? Does not seem used anywhere?
def find_continuum_pix(lambdas, spectra):
    """ Identify continuum pixels for use in normalization.

    Plotting function??

    f_bar (ensemble median at each pixel) and sigma_f (variance)
    Good cont pix have f_bar~1 and sigma_f<<1.

    Parameters
    ----------

    """
    import pylab as plt
    f_bar = np.median(spectra[:,:,0], axis=0)
    sigma_f = np.var(spectra[:,:,0], axis=0)
    # f_bar ~ 1...
    f_cut = 0.0001
    cont1 = np.abs(f_bar-1)/1 < f_cut
    # sigma_f << 1...
    sigma_cut = 0.005
    cont2 = sigma_f < sigma_cut
    cont = np.logical_and(cont1, cont2)
    plt.errorbar(lambdas[0][cont], f_bar[cont], yerr=sigma_f[cont], fmt='ko')


def continuum_normalize_Chebyshev(lambdas, fluxes, flux_errs, ivars):
    """Continuum-normalizes the spectra.

    Fit a 2nd order Chebyshev polynomial to each segment
    and divide each segment by its corresponding polynomial

    FIXME: code uses deg=3

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
        3D continuum-normalized spectra (nstars, npixels, 3)

    norm_ivar: ndarray
        normalized inverse variance

    continua: ndarray
            2D continuum array (nstars, npixels)
    """
    continua = np.zeros(lambdas.shape)
    norm_flux = np.zeros(fluxes.shape)
    norm_flux_err = np.zeros(flux_errs.shape)
    norm_ivar = np.zeros(ivars.shape)
    # list of "true" continuum pix, det. here by the Cannon
    # FIXME: HARD CODED filename
    pixlist = list(np.loadtxt("pixtest4.txt", dtype=int, usecols=(0,), unpack=1))

    # assigned and never used
    # ivars_orig = ivars

    contmask = np.ones(len(lambdas), dtype=bool)
    contmask[pixlist] = 0
    ivars[contmask] = 0.   # ignore non-cont pixels
    # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels

    # FIXME: HARD CODED values
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
