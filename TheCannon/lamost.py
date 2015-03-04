from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
from scipy import interpolate 
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
    Retrieves the data (assumse already in rest frame), creates and reads bad-
    pixel mask, builds inverse variance vectors, packages them all into 
    rectangular blocks.
    """
   
    def __init__(self, training_dir, test_dir, label_file):
        super(self.__class__, self).__init__(training_dir, test_dir, label_file)

    def _get_pixmask(self, file_in, middle, grid, flux, ivar):
        """ Return a mask array of bad pixels for one object's spectrum

        Bad pixels are defined as follows: fluxes or ivars are not finite, or 
        ivars are negative

        Major sky lines. 4046, 4358, 5460, 5577, 6300, 6363, 6863

        Where the red and blue wings join together: 5800-6000

        Read bad pix mask: file_in[0].data[4] is the ormask 

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
        npix = len(grid)
        
        bad_flux = (~np.isfinite(flux)) 
        bad_err = (~np.isfinite(ivar)) | (ivar <= 0)
        bad_pix_a = bad_err | bad_flux
        
        wings = np.logical_and(grid > 5750, grid < 6050)
        ormask = (file_in[0].data[4] > 0)[middle]
        bad_pix_b = wings | ormask

        max_pix_width = grid[npix-1]-grid[npix-2]
        skylines = np.array([4046, 4358, 5460, 5577, 6300, 6363, 6863])
        bad_pix_c = np.zeros(npix, dtype=bool)
        for skyline in skylines:
            badmin = skyline-max_pix_width
            badmax = skyline+max_pix_width
            bad_pix_temp = np.logical_and(grid > badmin, grid < badmax)
            bad_pix_c[bad_pix_temp] = True

        bad_pix_ab = bad_pix_a | bad_pix_b
        bad_pix = bad_pix_ab | bad_pix_c

        return bad_pix

    def _load_spectra(self, data_dir):
        """
        Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

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
        print("Loading spectra from directory %s" %data_dir)
        files = list(sorted([data_dir + "/" + filename
                 for filename in os.listdir(data_dir)]))
        nstars = len(files)  
        
        for jj, fits_file in enumerate(files):
            file_in = pyfits.open(fits_file)
            wl_temp = np.array(file_in[0].data[2])
            if jj == 0:
                npixels = len(wl_temp)
                SNRs = np.zeros(nstars, dtype=float)   
                start_wl = file_in[0].header['CRVAL1']
                diff_wl = file_in[0].header['CD1_1']
                val = diff_wl * (npixels) + start_wl
                grid_log = np.arange(start_wl,val, diff_wl)
                grid_log = np.delete(grid_log,-1)
                grid = np.array([10 ** aval for aval in grid_log])
                # get rid of edges
                middle = np.logical_and(grid > 4000, grid < 8800)
                grid = grid[middle]
                npixels = len(grid)
                fluxes = np.zeros((nstars, npixels), dtype=float)
                ivars = np.zeros(fluxes.shape, dtype=float)
            redshift = file_in[0].header['Z']
            wlshifts = redshift*wl_temp
            wl = wl_temp - wlshifts
            flux = np.array(file_in[0].data[0])
            ivar = np.array((file_in[0].data[1]))
            flux_rs = (interpolate.interp1d(wl, flux))(grid)
            ivar_rs = (interpolate.interp1d(wl, ivar))(grid)
            badpix = self._get_pixmask(file_in, middle, grid, flux_rs, ivar_rs)
            flux_rs = np.ma.array(flux_rs, mask=badpix)
            ivar_rs = np.ma.array(ivar_rs, mask=badpix)
            SNRs[jj] = np.ma.median(flux_rs*ivar_rs**0.5)
            ivar_rs = np.ma.filled(ivar_rs, fill_value=0.)
            fluxes[jj,:] = flux_rs
            ivars[jj,:] = ivar_rs

        print("Spectra loaded")
        return files, grid, fluxes, ivars, SNRs

