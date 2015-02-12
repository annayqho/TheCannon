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

class ApogeeDataset(Dataset):
    """ A class to represent a Dataset of APOGEE spectra and labels.

    Performs the APOGEE Munging necessary for making the data "Cannonizable." 
    Retrieves the data (assumse already in rest frame), creates and reads bad-
    pixel mask, builds inverse variance vectors, packages them all into 
    rectangular blocks.
    """
   
    def __init__(self, training_dir, test_dir, label_file):
        super(self.__class__, self).__init__(training_dir, test_dir, label_file)
        # we discard the edges of the fluxes: 10 A, corresponding to ~50 pix
        self.ranges = [[371,3192], [3697,5997], [6461,8255]]

    def _get_pixmask(self, fluxes, flux_errs):
        """ Return a mask array of bad pixels

        Bad pixels are defined as follows: fluxes or errors are not finite, or 
        reported errors are negative.

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
        bad_flux = (~np.isfinite(fluxes)) | (fluxes == 0)   
        bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
        bad_pix = bad_err | bad_flux

        return bad_pix

    def _load_spectra(self, data_dir):
        """
        Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

        Returns
        -------
        wl: numpy ndarray of length npixels
            rest-frame wavelength vector

        fluxes: numpy ndarray of shape (nstars, npixels)
            training set or test set pixel intensities

        ivars: numpy ndarray of shape (nstars, npixels)
            inverse variances, parallel to fluxes
            
        SNRs: numpy ndarray of length nstars
        """
        print("Loading spectra from directory %s" %data_dir)
        files = list(sorted([data_dir + "/" + filename
                 for filename in os.listdir(data_dir)]))
        nstars = len(files)  
        
        for jj, fits_file in enumerate(files):
            file_in = pyfits.open(fits_file)
            flux = np.array(file_in[1].data)
            if jj == 0:
                npixels = len(flux)
                SNRs = np.zeros(nstars, dtype=float)   
                fluxes = np.zeros((nstars, npixels), dtype=float)
                ivars = np.zeros(fluxes.shape, dtype=float)
                start_wl = file_in[1].header['CRVAL1']
                diff_wl = file_in[1].header['CDELT1']
                val = diff_wl * (npixels) + start_wl
                wl_full_log = np.arange(start_wl,val, diff_wl)
                wl_full = [10 ** aval for aval in wl_full_log]
                wl = np.array(wl_full)
            flux_err = np.array((file_in[2].data))
            badpix = self._get_pixmask(flux, flux_err)
            flux = np.ma.array(flux, mask=badpix)
            flux_err = np.ma.array(flux_err, mask=badpix)
            SNRs[jj] = np.ma.median(flux/flux_err)
            ones = np.ma.array(np.ones(npixels), mask=badpix)
            ivar = ones / flux_err**2
            ivar = np.ma.filled(ivar, fill_value=0.)
            fluxes[jj,:] = flux
            ivars[jj,:] = ivar

        print("Spectra loaded")
        return wl, fluxes, ivars, SNRs

