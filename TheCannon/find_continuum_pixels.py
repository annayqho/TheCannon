import numpy as np

LARGE = 200.
SMALL = 1. / LARGE

def _find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars):
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
    bad = np.logical_and(f_bar==0, sigma_f==0)
    cont1 = np.abs(f_bar-1) <= f_cut
    cont2 = sigma_f <= sig_cut
    contmask = np.logical_and(cont1, cont2)
    contmask[bad] = False
    return contmask


def _find_contpix(wl, fluxes, ivars, target_frac):
    """ Find continuum pix in spec, meeting a set target fraction

    Parameters
    ----------
    wl: numpy ndarray
        rest-frame wavelength vector

    fluxes: numpy ndarray
        pixel intensities
    
    ivars: numpy ndarray
        inverse variances, parallel to fluxes

    target_frac: float
        the fraction of pixels in spectrum desired to be continuum

    Returns
    -------
    contmask: boolean numpy ndarray
        True corresponds to continuum pixels
    """
    print("Target frac: %s" %(target_frac))
    bad1 = np.median(ivars, axis=0) == SMALL
    bad2 = np.var(ivars, axis=0) == 0
    bad = np.logical_and(bad1, bad2)
    npixels = len(wl)-sum(bad)
    f_cut = 0.0001
    stepsize = 0.0001
    sig_cut = 0.0001
    contmask = _find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars)
    if npixels > 0:
        frac = sum(contmask)/float(npixels)
    else:
        frac = 0
    while (frac < target_frac): 
        f_cut += stepsize
        sig_cut += stepsize
        contmask = _find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars)
        if npixels > 0:
            frac = sum(contmask)/float(npixels)
        else:
            frac = 0
    if frac > 0.10*npixels:
        print("Warning: Over 10% of pixels identified as continuum.")
    print("%s out of %s pixels identified as continuum" %(sum(contmask), 
                                                          npixels))
    print("Cuts: f_cut %s, sig_cut %s" %(f_cut, sig_cut))
    return contmask


def _find_contpix_regions(wl, fluxes, ivars, frac, ranges):
    """ Find continuum pix in a spectrum split into chunks

    Parameters
    ----------
    wl: numpy ndarray
        rest-frame wavelength vector

    fluxes: numpy ndarray
        pixel intensities

    ivars: numpy ndarray
        inverse variances, parallel to fluxes

    frac: float
        fraction of pixels in spectrum to be found as continuum

    ranges: list, array
        starts and ends indicating location of chunks in array

    Returns
    ------
    contmask: numpy ndarray, boolean
        True indicates continuum pixel
    """
    contmask = np.zeros(len(wl), dtype=bool)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        contmask[start:stop] = _find_contpix(
                wl[start:stop], fluxes[:,start:stop], ivars[:,start:stop], frac)
    return contmask
