""" Finds and returns list of continuum pixels, as a mask. """

def find_contpix(lambdas, fluxes, ivars):
    """ Find and return continuum pixels

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

