from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
import os
import sys
from cannon.helpers import Table
from cannon.dataset import Dataset
import matplotlib.pyplot as plt

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

class LamostDataset(Dataset):
    """ A class to represent a Dataset of LAMOST spectra and labels.

    Performs the LAMOST Munging necessary for making the data "Cannonizable."
    Retrieves the data (assumes already in rest frame), creates and reads bad-
    pixel mask, builds inverse variance vectors, packages them all into 
    rectangular blocks.
    """

    def __init__(self, training_dir, test_dir, label_file):
        super(self.__class__, self).__init__(training_dir, test_dir, label_file)

    def _get_pixmask(self, fluxes, flux_errs):
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
            ndarray giving bad pixels as True values
        """
        bad_flux = (~np.isfinite(fluxes)) 
        bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
        bad_pix = bad_err | bad_flux

        return bad_pix

    


