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
        
        bad_flux = (~np.isfinite(flux)) # count: 0
        bad_err = (~np.isfinite(ivar)) | (ivar <= 0)
        # ivar == 0 for approximately 3-5% of pixels
        bad_pix_a = bad_err | bad_flux
        
        # LAMOST people: wings join together, 5800-6000 Angstroms
        wings = np.logical_and(grid > 5800, grid < 6000)
        # this is another 3-4% of the spectrum
        # ormask = (file_in[0].data[4] > 0)[middle]
        # ^ problematic...this is over a third of the spectrum!
        # leave out for now
        # bad_pix_b = wings | ormask
        bad_pix_b = wings

        spread = 3 # due to redshift
        skylines = np.array([4046, 4358, 5460, 5577, 6300, 6363, 6863])
        bad_pix_c = np.zeros(npix, dtype=bool)
        for skyline in skylines:
            badmin = skyline-spread
            badmax = skyline+spread
            bad_pix_temp = np.logical_and(grid > badmin, grid < badmax)
            bad_pix_c[bad_pix_temp] = True
        # 34 pixels

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
        files = np.array(files)
        nstars = len(files)

        for jj, fits_file in enumerate(files):
            file_in = pyfits.open(fits_file)
            if jj == 0:
                # all stars start out on the same wavelength grid
                grid_all = np.array(file_in[0].data[2])
                # some spectra will end up with different pixels
                # because of the wavelength correction. so do this to ensure
                # that the interpolation never extrapolates...
                # values determined by experimentation, may change later
                middle = np.logical_and(grid_all > 3705, grid_all < 9091)
                # only lost 10 pixels here
                grid = grid_all[middle]
                npixels = len(grid) 
                SNRs = np.zeros(nstars, dtype=float)   
                fluxes = np.zeros((nstars, npixels), dtype=float)
                ivars = np.zeros(fluxes.shape, dtype=float)
            # correct for radial velocity of star
            redshift = file_in[0].header['Z']
            wlshift = redshift*grid_all
            wl = grid_all - wlshift
            flux = np.array(file_in[0].data[0])
            ivar = np.array((file_in[0].data[1]))
            # resample onto a common grid
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

