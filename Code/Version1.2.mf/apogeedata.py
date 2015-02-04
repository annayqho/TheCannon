from __future__ import (absolute_import, division, print_function,)
import numpy as np
import os
import sys
from cannon.helpers import Table
from cannon.dataset import DataFrame


# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange


try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits


def get_pixmask(fluxes, flux_errs):
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
    bad_flux = ~np.isfinite(fluxes)   
    bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
    bad_pix = bad_err | bad_flux

    return bad_pix


def continuum_normalize_Chebyshev(lambdas, fluxes, flux_errs, ivars,
                                  pixtest_fname, ranges, deg=3):
    """Continuum-normalizes the spectra.

    Fit a 2nd order Chebyshev polynomial to each segment
    and divide each segment by its corresponding polynomial

    Parameters
    ----------
    lambda: ndarray
        common wavelength

    fluxes: ndarray
        array of fluxes

    flux_errs: ndarray
        measurement uncertainties on fluxes

    ivars: ndarray
        inverse variance matrix

    pixtest_fname: str
        filename against which testing continuum subtraction

    ranges: sequence
        sequence of wavelength ranges defining the regions of interest

    Returns
    -------
    norm_flux: ndarray
        3D continuum-normalized spectra (nstars, npixels, 3)

    norm_ivar: ndarray
        normalized inverse variance

    continua: ndarray
            2D continuum array (nstars, npixels)
    """
    continua = np.zeros(lambdas.shape)
    norm_flux = np.zeros(fluxes.shape)
    norm_flux_err = np.zeros(flux_errs.shape)
    norm_ivar = np.zeros(ivars.shape)
    # list of "true" continuum pix, det. here by the Cannon
    pixlist = list(np.loadtxt(pixtest_fname, dtype=int, usecols=(0,), unpack=1))

    # assigned and never used
    # ivars_orig = ivars

    contmask = np.ones(len(lambdas), dtype=bool)
    contmask[pixlist] = 0
    ivars[contmask] = 0.   # ignore non-cont pixels
    # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels

    for i in range(len(ranges)):
        start, stop = ranges[i][0], ranges[i][1]
        flux = fluxes[start:stop]
        flux_err = flux_errs[start:stop]
        lambda_cut = lambdas[start:stop]
        ivar = ivars[start:stop]
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=lambda_cut, y=flux,
                                                    w=ivar, deg)
        continua[start:stop] = fit(lambda_cut)
        norm_flux[start:stop] = flux/fit(lambda_cut)
        norm_flux_err[start:stop] = flux_err/fit(lambda_cut)
        norm_ivar[start:stop] = 1. / norm_flux_err[start:stop]**2
    return norm_flux, norm_ivar, continua


class ApogeeDF(DataFrame):
    def __init__(self, spec_dir, label_file, contpix_file, pixtest_file):

        super(self.__class__, self).__init__(spec_dir, label_file, contpix_file)
        self.pixtest_file = pixtest_file
        self.ranges = [[371,3192], [3697,5997], [6461,8255]]

    def get_spectra(self, *args, **kwargs):
        """
        Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

        Returns
        -------
        lambdas: numpy ndarray of shape (npixels)
            common wavelength of the spectra

        spectra: 3D numpy ndarray of shape (nstars, npixels, 2)
            array of spectra expected with the following format:
                spectra[:,:,0] = flux values
                spectra[:,:,1] = flux err values

        large: float
            default dispersion value for bad data
        """
        files = [self.spec_dir + "/" + filename
                 for filename in os.listdir(self.spec_dir)]

        files = list(sorted(files))  # replaced by builtin

        LARGE = kwargs.get('large', 1000000.)
        nstars = len(files)   # moved outside the loop
        for jj, fits_file in enumerate(files):
            file_in = pyfits.open(fits_file)
            fluxes = np.array(file_in[1].data)
            if jj == 0:
                npixels = len(fluxes)
                SNRs = np.zeros(nstars, dtype=float)   # imposing the type is faster
                norm_fluxes = np.zeros((nstars, npixels), dtype=float)
                norm_ivars = np.zeros(norm_fluxes.shape, dtype=float)
                start_wl = file_in[1].header['CRVAL1']
                diff_wl = file_in[1].header['CDELT1']
                val = diff_wl * (npixels) + start_wl
                wl_full_log = np.arange(start_wl,val, diff_wl)
                wl_full = [10 ** aval for aval in wl_full_log]
                lambdas = np.array(wl_full)
            flux_errs = np.array((file_in[2].data))
            badpix = get_pixmask(fluxes, flux_errs)
            fluxes = np.ma.array(fluxes, mask=badpix, fill_value=0.)
            flux_errs = np.ma.array(flux_errs, mask=badpix, fill_value=LARGE)
            SNRs[jj] = np.ma.median(fluxes/flux_errs)
            ones = np.ma.array(np.ones(npixels), mask=badpix)
            fluxes = np.ma.filled(fluxes)
            flux_errs = np.ma.filled(flux_errs)
            ivar = ones / (flux_errs**2)
            ivar = np.ma.filled(ivar, fill_value=0.)
            norm_flux, norm_ivar, continua = \
                continuum_normalize_Chebyshev(lambdas, fluxes, flux_errs, ivar,
                                              self.pixtest_file, self.ranges)
            # check null division
            ind = np.isfinite(norm_ivar) & (norm_ivar > 0)
            norm_ivar[~ind] = norm_ivar[ind].min() * 1e-2
            badpix2 = get_pixmask(norm_flux, 1. / np.sqrt(norm_ivar))
            temp = np.ma.array(norm_flux, mask=badpix2, fill_value=1.0)
            norm_fluxes[jj] = np.ma.filled(temp)
            temp = np.ma.array(norm_ivar, mask=badpix2, fill_value=0.)
            norm_ivars[jj] = np.ma.filled(temp)

        # len gives an int and you print a string
        # print("Loaded %s stellar spectra" %len(files))
        # new formatting is more flexible
        print("Loaded {0:d} stellar spectra".format(len(files)))
        return lambdas, norm_fluxes, norm_ivars, SNRs

    def get_reference_labels(self, *args, **kwags):
        """Extracts training labels from file.

        Assumes that the file has # then label names in first row, that first
        column is the filename, that the remaining values are floats
        and that you want all of the labels. User picks specific labels later.

        Input: name(string) of the data file containing the labels
        Returns: label_names list, and np ndarray (size=numtrainingstars, nlabels)
        consisting of all of the label values
        """
        data = Table(self.label_file)
        data.sort('id')
        label_names = data.keys()
        nlabels = len(label_names)

        print("Loaded stellar IDs, format: %s" % data['id'][0])
        print("Loaded %d labels:" % nlabels)
        print(label_names)
        return data['id'], label_names, data
