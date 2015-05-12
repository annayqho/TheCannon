import numpy as np

LARGE = 200.
SMALL = 1. / LARGE

""" Finds and returns list of continuum pixels, as a mask. """

def find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars):
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
    #bad1 = np.median(ivars, axis=0) < SMALL
    bad1 = np.median(fluxes, axis=0) == 0.
    bad2 = np.var(fluxes, axis=0) == 0.
    #bad2 = np.var(ivars, axis=0) == 0
    bad = np.logical_and(bad1, bad2)
    f_bar = np.median(fluxes, axis=0)
    sigma_f = np.var(fluxes, axis=0)
    f_bar = np.ma.array(f_bar, mask=bad)
    sigma_f = np.ma.array(sigma_f, mask=bad)
    cont1 = np.abs(f_bar-1) <= f_cut
    cont2 = sigma_f <= sig_cut
    # cont3 = sigma_f >= np.abs(1-f_bar)
    contmask1 = np.logical_and(cont1, cont2)
    # contmask = np.logical_and(contmask1, cont3)
    contmask1 = np.ma.filled(contmask1, fill_value=False)
    return contmask1

def find_contpix(wl, fluxes, ivars, target_frac):
    print("Target frac: %s" %(target_frac))
    bad1 = np.median(ivars, axis=0) == SMALL
    bad2 = np.var(ivars, axis=0) == 0
    bad = np.logical_and(bad1, bad2)
    npixels = len(wl)-sum(bad)
    f_cut = 0.0001
    stepsize = 0.0001
    sig_cut = 0.0001
    contmask = find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars)
    if npixels > 0:
        frac = sum(contmask)/float(npixels)
    else:
        frac = 0
    while (frac < target_frac): 
        f_cut += stepsize
        sig_cut += stepsize
        contmask = find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars)
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

def find_contpix_regions(wl, fluxes, ivars, frac, ranges):
    print("taking spectra in %s regions" %len(ranges))
    contmask = np.zeros(len(wl), dtype=bool)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        contmask[start:stop] = find_contpix(
                wl[start:stop], fluxes[:,start:stop], ivars[:,start:stop], frac)
    return contmask
