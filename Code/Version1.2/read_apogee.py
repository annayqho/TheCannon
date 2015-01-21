"""Extract & continuum-normalize spectra from APOGEE .fits files."""

import pyfits
import numpy as np
import os
import matplotlib.pyplot as plt

def get_spectra(files):
    """
    Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

    Parameters
    ----------
    a list of data file names of length nstars
    
    Returns
    -------
    lambdas: numpy ndarray of shape (npixels)
    spectra: 2D numpy ndarray of shape (nstars, npixels, 2)
    with spectra[:,:,0] = flux values
    spectra[:,:,1] = flux err values
    """
    
    nstars = len(files)
    for jj,fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        fluxes = np.array(file_in[1].data)
        if jj == 0: 
            npixels = len(fluxes)
            SNRs = np.zeros(nstars)
            lambdas = np.zeros(npixels)
            spectra = np.zeros((nstars, npixels, 2))
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl*(npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10**aval for aval in wl_full_log]
            lambdas = np.array(wl_full)
        flux_errs = np.array((file_in[2].data))
        SNRs[jj] = float(file_in[0].header['SNR'])
        spectra[jj, :, 0] = fluxes
        spectra[jj, :, 1] = flux_errs
    print "Loaded %s stellar spectra" %nstars
   
    # Deal with bad pixels
    for jj in range(nstars):
        bad1 = np.logical_or(np.isinf(spectra[jj,:,0]), 
                np.isinf(spectra[jj,:,1]))
        bad = np.logical_or(bad1, spectra[jj,:,1] <= 0)
        print sum(bad)

    # Continuum normalize
    contpix = identify_continuum(lambdas, spectra)
    normalized_spectra, continua = continuum_normalize(lambdas, spectra, contpix)
    return lambdas, normalized_spectra, continua, SNRs

def identify_continuum(lambdas, spectra):
    """Identifies continuum pixels."""
    
    f_bar = np.median(spectra[:,:,0], axis=0)
    sigma_f = np.var(spectra[:,:,0], axis=0)
    # f_bar == 0
    cont1 = f_bar == 0
    # f_bar ~ 1...
    f_cut = 0.001
    cont2 = np.abs(f_bar-1)/1 < f_cut
    # sigma_f << 1...
    sigma_cut = 0.005
    cont3 = sigma_f < sigma_cut
    cont = np.logical_or(cont1, np.logical_and(cont2, cont3))
    #plot(lambdas, f_bar)
    #errorbar(lambdas[cont], f_bar[cont], yerr=sigma_f[cont], fmt='ko')
    return lambdas[cont] 

def continuum_normalize_Chebyshev(lambdas, spectra, contpix):
    """Continuum-normalizes the spectra.

    Fit a 2nd order Chebyshev polynomial to each segment 
    and divide each segment by its corresponding polynomial 

    Input: lambdas, numpy ndarray floats
        spectra array, 2D float shape nstars,npixels,3
        contpix, identified continuum pixels
    Returns: 3D continuum-normalized spectra (nstars, npixels,3)
            2D continuum array (nstars, npixels)
    """
    nstars = spectra.shape[0]
    npixels = len(lambdas)
    continua = np.zeros((nstars, npixels))
    normalized_spectra = np.ones((nstars, npixels, 2))
    # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels
    ranges = [[371,3192], [3697,5997], [6461,8255]]
    for jj in range(nstars):
        var_array = 100**2*np.ones(npixels)
        var_array[contpix] = 0.000
        ivar = 1. / ((spectra[jj,:,1]**2) + var_array)
        ivar = (np.ma.masked_invalid(ivar)).filled(0)
        for i in range(len(ranges)):
            start, stop = ranges[i][0], ranges[i][1]
            spectrum = spectra[jj,start:stop,:]
            lambda_cut = lambdas[start:stop]
            ivar1 = ivar[start:stop]
            fit = np.polynomial.chebyshev.Chebyshev.fit(x=lambda_cut, 
                    y=spectrum[:,0], w=ivar1, deg=3)
            continua[jj,start:stop] = fit(lambda_cut)
            normalized_spectra[jj,start:stop,0] = spectrum[:,0]/fit(lambda_cut)
            normalized_spectra[jj,start:stop,1] = spectrum[:,1]/fit(lambda_cut)

