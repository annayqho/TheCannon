""" Finds and returns list of continuum pixels, as a mask. """

def get_num_contpix(f_cut, sig_cut, lambdas, fluxes, ivars):
    """ Find and return continuum pixels given the flux and sigma cut

    Parameters
    ----------
    f_cut: float
        the upper limit imposed on the quantity (fbar-1)
    sig_cut: float
        the upper limit imposed on the quantity (f_sig)
    lambdas: numpy ndarray of length npixels
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
    cont1 = np.abs(f_bar-1) < f_cut
    cont2 = sigma_f < sigma_cut
    contmask = np.logical_and(cont1, cont2)
    return contmask

def find_contpix(lambdas, fluxes, ivars):
    """ Find and return continuum pixels

    Searches through some flux and sigma cuts such that between 5 and 10%
    of pixels are identified as continuum.

    Parameters
    ----------
    lambdas: numpy ndarray of length npixels
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
    npixels = len(lambdas)
    f_cut = 0.003
    stepsize = 0.0001
    sig_cut = 0.003
    contmask = get_num_contpix(f_cut, sig_cut, lambdas, fluxes, ivars)
    frac = sum(contmask)/npixels
    while frac < 0.05*npixels: 
        f_cut += sig_step
        contmask = get_num_contpix(f_cut, sig_cut, lambdas, fluxes, ivars)
        frac = sum(contmask)/npixels
    if frac > 0.10*npixels:
        print("Warning: Over 10% of pixels identified as continuum.")
    print("%s out of %s pixels identified as continuum" %(sum(contmask), 
                                                          npixels))
    return contmask
