import numpy as np

""" Finds and returns list of continuum pixels, as a mask. """

def find_contpix_cuts(f_cut, sig_cut, wl, fluxes, ivars):
    """ Find and return continuum pixels given the flux and sigma cut

    Parameters
    ----------
    f_cut: float
        the upper limit imposed on the quantity (fbar-1)
    sig_cut: float
        the upper limit imposed on the quantity (f_sig)
    wl: numpy ndarray of length npixels
        rest-frame wavelength vector
    fluxes: numpy ndarray of shape (nstars, npixels)
        pixel intensities
    ivars: numpy ndarray of shape nstars, npixels
        inverse variances, parallel to fluxes

    Returns
    -------
    contmask: boolean mask of length npixels
        True indicates that the pixel is continuum
    """
    f_bar = np.median(fluxes, axis=0)
    sigma_f = np.var(fluxes, axis=0)
    cont1 = np.abs(f_bar-1) <= f_cut
    cont2 = sigma_f <= sig_cut
    cont3 = sigma_f >= np.abs(1-f_bar)
    contmask1 = np.logical_and(cont1, cont2)
    contmask = np.logical_and(contmask1, cont3)
    return contmask

def find_contpix(wl, fluxes, ivars):
    """ Find and return continuum pixels

    Searches through some flux and sigma cuts such that between 5 and 10%
    of pixels are identified as continuum.

    Parameters
    ----------
    wl: numpy ndarray of length npixels
        rest-frame wavelength vector
    fluxes: numpy ndarray of shape (nstars, pixels)
        pixel intensities
    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes

    Returns
    -------
    contmask: boolean mask of length npixels
        True indicates that the pixel is continuum
    """
    # Apply a cut based on the median and variance vector
    # Adjust the cut levels so that 5-7% of pixels are identified as continuum
    bad1 = np.median(ivars, axis=0) == 0
    bad2 = np.var(ivars, axis=0) == 0
    bad = np.logical_and(bad1, bad2)
    npixels = len(wl)-sum(bad)
    f_cut = 0.003
    stepsize = 0.0001
    sig_cut = 0.003
    contmask = find_contpix_cuts(f_cut, sig_cut, wl, fluxes, ivars)
    frac = sum(contmask)/float(npixels)
    while frac < 0.05: 
        f_cut += stepsize
        sig_cut += stepsize
        contmask = find_contpix_cuts(f_cut, sig_cut, wl, fluxes, ivars)
        frac = sum(contmask)/float(npixels)
    if frac > 0.10*npixels:
        print("Warning: Over 10% of pixels identified as continuum.")
    print("%s out of %s pixels identified as continuum" %(sum(contmask), 
                                                          npixels))
    return contmask
