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
    
    for jj,fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        fluxes = np.array(file_in[1].data)
        if jj == 0: 
            nstars = len(files) 
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
    print "Loaded %s stellar spectra" %len(files)
    
    # Automatically continuum-normalize
    normalized_spectra, continua = continuum_normalize_Chebyshev(spectra)
    return lambdas, normalized_spectra, continua, SNRs

def continuum_normalize(lambdas, spectra):
    """Continuum-normalizes the spectra.

    Create two vectors that contain a) f_bar the ensemble median (or
    slightly higher quantile) at each pixel, taken over the set of
    reference objects; b) sigma_f: the variance (formal second moment,
    interval enclosing the central X% of the distribution at each 
    pixel; X=68 to 90). Good continuum pixels are those that have 
    f_bar~1 and sigma_f<<1. Then an (initially by eye) cut on f_bar 
    and sigma_f may identify (in a very simple fashion) good cont
    pixels. Drawbacks: assuming label space is covered, but unevenly,
    then what you do for sigma_f matters."""
    
    f_bar = np.median(spectra[:,:,0], axis=0)
    sigma_f = np.var(spectra[:,:,0], axis=0)
    # f_bar ~ 1...
    f_cut = 0.0001
    cont1 = np.abs(f_bar-1)/1 < f_cut
    # sigma_f << 1...
    sigma_cut = 0.005
    cont2 = sigma_f < sigma_cut
    cont = np.logical_and(cont1, cont2)
    errorbar(lambdas[0][cont], f_bar[cont], yerr=sigma_f[cont], fmt='ko')

def continuum_normalize_Chebyshev(lambdas, spectra):
    """Continuum-normalizes the spectra.

    Fit a 2nd order Chebyshev polynomial to each segment 
    and divide each segment by its corresponding polynomial 

    Input: spectra array, 2D float shape nstars,npixels,3
    Returns: 3D continuum-normalized spectra (nstars, npixels,3)
            2D continuum array (nstars, npixels)
    """
    nstars = spectra.shape[0]
    npixels = len(lambdas)
    continua = np.zeros((nstars, npixels))
    normalized_spectra = np.ones((nstars, npixels, 2))
    # list of "true" continuum pix, det. here by the Cannon
    pixlist = list(np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1))
    # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels
    ## I found the regions with flux to be: 
    ## [321, 3242] [3647, 6047], [6411, 8305]
    ## With edge cuts: [371, 3192], [3697, 5997], [6461, 8255]
    ## Corresponding to: [15218, 15743] [15931, 16367] [16549, 16887] 
    ranges = [[371,3192], [3697,5997], [6461,8255]]
    LARGE = 200.
    for jj in range(nstars):
        # Mask unphysical pixels
        bad1 = np.invert(np.logical_and(np.isfinite(spectra[jj,:,0]),  
            np.isfinite(spectra[jj,:,1])))
        bad = bad1 | (spectra[jj,:,1] <= 0)
        spectra[jj,:,0][bad] = 0.
        spectra[jj,:,1][bad] = np.Inf
        var_array = 100**2*np.ones(npixels)
        var_array[pixlist] = 0.000
        ivar = 1. / ((spectra[jj, :, 1] ** 2) + var_array)
        ivar = (np.ma.masked_invalid(ivar)).filled(0)
        for i in range(len(ranges)):
            start, stop = ranges[i][0], ranges[i][1]
            spectrum = spectra[jj,start:stop,:]
            ivar1 = ivar[start:stop]
            fit = np.polynomial.chebyshev.Chebyshev.fit(x=lambdas, 
                    y=spectrum[:,0], w=ivar1, deg=3)
            continua[jj,start:stop] = fit(lambdas)
            normalized_fluxes = spectrum[:,0]/fit(lambdas[start:stop])
            bad = np.invert(np.isfinite(normalized_fluxes))
            normalized_fluxes[bad] = 1.
            normalized_flux_errs = spectrum[:,1]/fit(lambdas[start:stop])
            bad = np.logical_or(np.invert(np.isfinite(normalized_flux_errs)),
                    normalized_flux_errs <= 0)
            normalized_flux_errs[bad] = LARGE
            normalized_spectra[jj,start:stop,0] = normalized_fluxes 
            normalized_spectra[jj,start:stop,1] = normalized_flux_errs
        # One last check for unphysical pixels
        bad = spectra[jj,:,1] > LARGE
        normalized_spectra[jj,np.logical_or(bad, bad1),0] = 1.
        normalized_spectra[jj,np.logical_or(bad, bad1),1] = LARGE
    return normalized_spectra, continua
