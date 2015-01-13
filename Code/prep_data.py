import numpy as np
import pyfits
import os

"""Provides basic optional methods for extracting spectra and labels from 
a data file.

Makes the following assumptions: the file with spectra is aspcap .fits,
the spectrum is continuous
the file with training labels is...

Methods
-------
get_spectra
continuum_normalize
get_training_labels
"""

def get_spectra(filenames):
    """Extracts spectra from aspcap fits files

    Input: a list of data file names 
    Returns: 3D float array 
    spectra[:,:,0] = pixel wavelengths
    spectra[:,:,1] = flux values
    spectra[:,:,2] = flux err values
    """

    for jj,fits_file in enumerate(filenames):
        file_in = pyfits.open(fits_file)
        fluxes = np.array(file_in[1].data)
        if jj == 0:
            nstars = len(filenames)
            npixels = len(fluxes)
            spectra = np.zeros((nstars, npixels, 3))
            print "Constructing spectra array, shape (nstars, npixels, 3)"
            print "= (%s, %s, 3)" %(nstars, npixels)
        flux_errs = np.array((file_in[2].data))
        start_wl = file_in[1].header['CRVAL1']
        diff_wl = file_in[1].header['CDELT1']
        val = diff_wl*(npixels) + start_wl
        wl_full_log = np.arange(start_wl,val, diff_wl)
        wl_full = [10**aval for aval in wl_full_log]
        pixels = np.array(wl_full)
        spectra[jj, :, 0] = pixels
        spectra[jj, :, 1] = fluxes
        spectra[jj, :, 2] = flux_errs
    return spectra

# Need to discuss: what about this should we allow the user to customize?
def continuum_normalize(spectra):
    """Continuum-normalizes a spectra array.

    Assumes that there are no gaps in the spectrum and fits a 2nd order
    Chebyshev polynomial. Divides spectrum by the polynomial.

    Input: spectra array, 2D float shape nstars,npixels,3
    Returns: 3D continuum-normalized spectra (nstars, npixels,3)
    2D continuum array (nstars, npixels)
    """

    nstars = spectra.shape[0]
    npixels = spectra.shape[1]
    continua = np.zeros((nstars, npixels))
    normalized_spectra = np.ones((nstars, npixels, 3))
    ### NEED TO write a method to det. these "true" continuum pixels
    pixlist = list(np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1))
    # Options to discard the edges of the flux, say 10 Angstroms (50 pixels)?
    # Options to find gaps in the spectrum and split it up?
    LARGE = 200.
    for jj in range(nstars):
        # Mask unphysical pixels
        bad1 = np.invert(np.logical_and(np.isfinite(spectra[jj,:,1]),
                         np.isfinite(spectra[jj,:,2])))
        bad = bad1 | (spectra[jj,:,2] <= 0)
        spectra[jj,:,1][bad] = 0.
        spectra[jj,:,2][bad] = np.Inf
        var_array = 100**2*np.ones(npixels)
        var_array[pixlist] = 0.000
        ivar = 1. / ((spectra[jj, :, 2] ** 2) + var_array)
        ivar = (np.ma.masked_invalid(ivar)).filled(0)
        spectrum = spectra[jj,:,:]
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=spectrum[:,0],
                y=spectrum[:,1], w=ivar, deg=3)
        continua[jj,:] = fit(spectrum[:,0])
        normalized_fluxes = spectrum[:,1]/fit(spectra[0,:,0])
        bad = np.invert(np.isfinite(normalized_fluxes))
        normalized_fluxes[bad] = 1.
        normalized_flux_errs = spectrum[:,2]/fit(spectra[0,:,0])
        bad = np.logical_or(np.invert(np.isfinite(normalized_flux_errs)),
                normalized_flux_errs <= 0)
        normalized_flux_errs[bad] = LARGE
        normalized_spectra[jj,:,0] = spectra[jj,:,0]
        normalized_spectra[jj,:,1] = normalized_fluxes
        normalized_spectra[jj,:,2] = normalized_flux_errs
        # One last check for unphysical pixels
        bad = spectra[jj,:,2] > LARGE
        normalized_spectra[jj,np.logical_or(bad, bad1),1] = 1.
        normalized_spectra[jj,np.logical_or(bad, bad1),2] = LARGE
    return normalized_spectra, continua

def get_training_labels(filename):
    """Extracts training label names and values from file

    Assumes:
    -- that the file format is as follows, with label names in first row
    -- first row has a # then label names...
    -- first column is the IDs...so strings
    -- and the rest are floats
    -- that you want *all* of the labels

    Input: filename
    Returns: 2D np.array (size=ntrainingstars, nlabels) consisting of all
    training labels
    """

    with open(filename, 'r') as f:
        all_labels = f.readline().split()[1:] # ignore the hash
    ID_type = all_labels[0]
    label_names = all_labels[1:]
    print "Saving stellar IDs, %s" %ID_type
    IDs = np.loadtxt(filename, usecols = (0,), dtype='string')
    nlabels = len(label_names)
    print "Loading %s labels:" %nlabels
    print label_names
    cols = tuple(xrange(1,nlabels+1))
    label_values = np.loadtxt(filename, usecols=cols)
    return label_names, label_values 
