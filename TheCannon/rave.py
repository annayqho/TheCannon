from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
from scipy import interpolate 
import os
import sys
from cannon.helpers import Table
from cannon.dataset import Dataset
import matplotlib.pyplot as plt
from scipy.io.idl import readsav

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

class RaveDataset(Dataset):
    """ A class to represent a Dataset of LAMOST spectra and labels.

    Performs the LAMOST Munging necessary for making the data "Cannonizable." 
    Retrieves the data (assumse already in rest frame), creates and reads bad-
    pixel mask, builds inverse variance vectors, packages them all into 
    rectangular blocks.
    """
   
    def __init__(self, data_dir, tr_list, test_list, label_file):
        super(self.__class__, self).__init__(data_dir, tr_list, test_list, label_file)
        self.ranges = None

    def _get_pixmask(self, file_in, wl, middle, flux, ivar):
        """ Return a mask array of bad pixels for one object's spectrum

        Parameters
        ----------
        fluxes: ndarray
            flux array

        flux_errs: ndarray
            measurement uncertainties on fluxes

        Returns
        -------
        mask: ndarray, dtype=bool
            array giving bad pixels as True values
        """
        npix = len(wl)
        bad_pix = np.zeros(npix, dtype=bool)
        return bad_pix

    def _load_spectra(self, data_dir, filenames):
        """
        Extracts spectra (wavelengths, fluxes, fluxerrs) from rave files

        Returns
        -------
        IDs: list of length nstars
            stellar IDs
        
        wl: numpy ndarray of length npixels
            rest-frame wavelength vector

        fluxes: numpy ndarray of shape (nstars, npixels)
            training set or test set pixel intensities

        ivars: numpy ndarray of shape (nstars, npixels)
            inverse variances, parallel to fluxes
            
        SNRs: numpy ndarray of length nstars
        """ 
        data = readsav('RAVE_DR4_calibration_data.save')
        items = data.items()


        print("Spectra loaded")
        return files, grid, fluxes, ivars, SNRs

