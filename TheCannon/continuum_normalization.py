import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize as opt

LARGE = 200.
SMALL = 1. / LARGE


def _partial_func(func, *args, **kwargs):
    """ something """
    def wrap(x, *p):
        return func(x, p, **kwargs)
    return wrap


def gaussian_weight(wl_i, wl_0, L):
    """ The weight of a pixel i given a Gaussian centered on pixel 0 

    Parameters
    ----------
    lambdai: float
        the pixel of interest
    lambda0: float
        the center of the Gaussian
    L: float
        the width of the Gaussian

    Returns
    -------
    the weight of pixel i
    """
    return np.exp[-0.5*(wl_i-wl_0)**2/L**2]


def smoothed_spectrum_single_pix(wl_0, wl, flux, ivar, L):
    """ Returns the weighted mean flux for a particular pixel

    Parameters
    ----------
    wl_0: float
        the wavelength of the center of the Gaussian
    wl: numpy ndarray
        wavelengths of pixels in spectrum
    flux: numpy ndarray 
        flux values of spectrum
    ivar: numpy ndarray
        ivar values of spectrum
    L: float
        the width of the Gaussian
    
    Returns
    -------
    the smoothed mean flux value
    """
    num = 0
    den = 0
    for ii in range(0, npix):
        weight = gaussian_weight(wl_ii, wl_0, L)
        num += weight*ivar_ii*flux_ii
        den += weight*ivar_ii
    return num/den


def smoothed_spectrum(wl, flux, ivar, L):
    """ Returns the weighted mean spectrum

    Parameters
    ----------
    wl: numpy ndarray
        wavelengths
    flux: numpy ndarray
        flux values of spectrum
    ivar: numpy ndarray
        inverse variances corresponding to flux values
    L: float
        width of Gaussian used to assign weights

    Returns
    -------
    smoothed_flux: numpy ndarray
        smoothed flux values, the mean spectrum
    """
    npix = len(wl)
    smoothed_flux = np.zeros(npix)
    for ii in range(npix):
        smoothed_flux[ii] = smoothed_spectrum_single_pix(wl[ii], wl, flux, ivar, L)
    return smoothed_flux


def smoothed_spectra(wl, fluxes, ivars, L):
    """ Returns the weighted mean block of spectra

    Parameters
    ----------
    wl: numpy ndarray
        wavelength vector
    flux: numpy ndarray
        block of flux values 
    ivar: numpy ndarray
        block of ivar values
    L: float
        width of Gaussian used to assign weights

    Returns
    -------
    smoothed_fluxes: numpy ndarray
        block of smoothed flux values, mean spectra
    """
    nstars = flux.shape[0]
    smoothed_fluxes = np.zeros(fluxes.shape)
    for ii in range(nstars):
        flux = fluxes[ii,:]
        ivar = ivars[ii,:]
        smoothed_fluxes[ii,:] = smoothed_spectrum(wl, flux, ivar, L)
    return smoothed_fluxes


def cont_norm_gaussian_smoothing(dataset, L):
    """ Continuum normalize by dividing by a Gaussian-weighted smoothed spectrum

    Parameters
    ----------
    dataset: Dataset
        the dataset to continuum normalize
    L: float
        the width of the Gaussian used for weighting

    Returns
    -------
    dataset: Dataset
        updated dataset
    """
    smoothed_tr_fluxes = smoothed_spectra(
            dataset.wl, dataset.tr_flux, dataset.test_ivar, L)
    smoothed_test_fluxes = smoothed_spectra(
            dataset.wl, dataset.test_flux, dataset.test_ivar, L)
    norm_tr_fluxes = dataset.tr_flux / smoothed_tr_fluxes 
    norm_test_fluxes = dataset.test_flux / smoothed_test_fluxes
    dataset.tr_flux = norm_tr_fluxes
    dataset.test_flux = norm_test_fluxes
    return dataset


def _cont_sinusoid(x, p, L, y):
    """ Return the sinusoid cont func evaluated at input x for the continuum.

    Parameters
    ----------
    x: float or np.array
        data, input to function
    p: ndarray
        coefficients of fitting function
    L: float
        width of x data 
    y: float or np.array
        output data corresponding to input x

    Returns
    -------
    func: float
        function evaluated for the input x
    """
    N = int(len(p)/2)
    n = np.linspace(0, N, N+1)
    k = n*np.pi/L
    func = 0
    for n in range(0, N):
        func += p[2*n]*np.sin(k[n]*x)+p[2*n+1]*np.cos(k[n]*x)
    return func


def _fit_cont(fluxes, ivars, contmask, deg, ffunc):
    """ Fit a continuum to a continuum pixels in a segment of spectra

    Functional form can be either sinusoid or chebyshev, with specified degree

    Parameters
    ----------
    fluxes: numpy ndarray of shape (nstars, npixels)
        training set or test set pixel intensities

    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes

    contmask: numpy ndarray of length (npixels)
        boolean pixel mask, True indicates that pixel is continuum 

    deg: int
        degree of fitting function

    ffunc: str
        type of fitting function, chebyshev or sinusoid

    Returns
    -------
    cont: numpy ndarray of shape (nstars, npixels)
        the continuum, parallel to fluxes
    """
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    cont = np.zeros(fluxes.shape)
    for jj in range(nstars):
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        pix = np.arange(0, npixels)
        y = flux[contmask]
        x = pix[contmask]
        yivar = ivar[contmask]
        yivar[yivar == 0] = SMALL**2   
        if ffunc=="sinusoid": 
            p0 = np.ones(deg*2) # one for cos, one for sin
            L = max(x)-min(x)
            pcont_func = _partial_func(_cont_sinusoid, L=L, y=flux)
            popt, pcov = opt.curve_fit(pcont_func, x, y, p0=p0, 
                                       sigma=1./np.sqrt(yivar))
        elif ffunc=="chebyshev":
            fit = np.polynomial.chebyshev.Chebyshev.fit(x=x,y=y,w=yivar,deg=deg)
        for element in pix:
            if ffunc=="sinusoid":
                cont[jj,element] = _cont_sinusoid(element, popt, L=L, y=flux)
            elif ffunc=="chebyshev":
                cont[jj,element] = fit(element)
    return cont


def _fit_cont_regions(fluxes, ivars, contmask, deg, ranges, ffunc):
    """ Run fit_cont, dealing with spectrum in regions or chunks

    This is useful if a spectrum has gaps.

    Parameters
    ----------
    fluxes: ndarray of shape (nstars, npixels)
        training set or test set pixel intensities

    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes

    contmask: numpy ndarray of length (npixels)
        boolean pixel mask, True indicates that pixel is continuum 

    deg: int
        degree of fitting function

    ffunc: str
        type of fitting function, chebyshev or sinusoid

    Returns
    -------
    cont: numpy ndarray of shape (nstars, npixels)
        the continuum, parallel to fluxes
    """
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    cont = np.zeros(fluxes.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        if ffunc=="chebyshev":
            output = _fit_cont(fluxes[:,start:stop],
                              ivars[:,start:stop],
                              contmask[start:stop], deg=deg, ffunc="chebyshev")
        elif ffunc=="sinusoid":
            output = _fit_cont(fluxes[:,start:stop],
                              ivars[:,start:stop],
                              contmask[start:stop], deg=deg, ffunc="sinusoid")
        cont[:,start:stop] = output
    return cont


def _weighted_median(values, weights, quantile):
    """ Calculate a weighted median for values above a particular quantile cut

    Used in pseudo continuum normalization

    Parameters
    ----------
    values: np ndarray of floats
        the values to take the median of
    weights: np ndarray of floats
        the weights associated with the values
    quantile: float
        the cut applied to the input data

    Returns
    ------
    the weighted median
    """

    sindx = np.argsort(values)
    cvalues = 1. * np.cumsum(weights[sindx])
    cvalues = cvalues / cvalues[-1]
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    indx = foo[0]
    return values[indx]


def _cont_norm_q(wl, fluxes, ivars, q, delta_lambda):
    """ Perform continuum normalization using a running quantile

    Parameters
    ----------
    wl: numpy ndarray 
        wavelength vector
    fluxes: numpy ndarray of shape (nstars, npixels)
        pixel intensities
    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes
    q: float
        the desired quantile cut
    delta_lambda: int
        the number of pixels over which the median is calculated

    Output
    ------
    norm_fluxes: numpy ndarray of shape (nstars, npixels)
        normalized pixel intensities
    norm_ivars: numpy ndarray of shape (nstars, npixels)
        rescaled pixel invariances
    """
    print("contnorm.py: continuum norm using running quantile")
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    cont = np.zeros(fluxes.shape)
    nstars = fluxes.shape[0]
    for jj in range(nstars):
        print("cont_norm_q(): working on star %s" %jj) 
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        for ll, lam in enumerate(wl):
            indx = (np.where(abs(wl-lam) < delta_lambda))[0]
            flux_cut = flux[indx]
            ivar_cut = ivar[indx]
            cont[jj,ll] = _weighted_median(flux_cut, ivar_cut, q)
    for jj in range(nstars):
        norm_fluxes[jj,:] = fluxes[jj,:]/cont[jj,:]
        norm_ivars[jj,:] = cont[jj,:]**2 * ivars[jj,:]
    return norm_fluxes, norm_ivars


def _cont_norm_q_regions(wl, fluxes, ivars, q, delta_lambda, ranges):
    """ Perform continuum normalization using running quantile, for spectrum
    that comes in chunks
    """
    print("contnorm.py: continuum norm using running quantile")
    print("Taking spectra in %s chunks" %len(ranges))
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm_q(wl[start:stop], fluxes[:,start:stop],
                             ivars[:,start:stop],
                             q, delta_lambda)
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    return norm_fluxes, norm_ivars


def _cont_norm(fluxes, ivars, cont):
    """ Continuum-normalize a continuous segment of spectra.

    Parameters
    ----------
    fluxes: numpy ndarray 
        pixel intensities
    ivars: numpy ndarray 
        inverse variances, parallel to fluxes
    contmask: boolean mask
        True indicates that pixel is continuum

    Returns
    -------
    norm_fluxes: numpy ndarray
        normalized pixel intensities
    norm_ivars: numpy ndarray
        rescaled inverse variances
    """
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for jj in range(nstars):
        bad = (ivars[jj,:] == 0)
        norm_fluxes[jj,:] = fluxes[jj,:]/cont[jj,:]
        norm_ivars[jj,:] = cont[jj,:]**2 * ivars[jj,:]
        norm_fluxes[jj,:][bad] = 1.
        norm_ivars[jj,:][bad] = SMALL**2
    return norm_fluxes, norm_ivars 


def _cont_norm_regions(fluxes, ivars, cont, ranges):
    """ Perform continuum normalization for spectra in chunks

    Useful for spectra that have gaps

    Parameters
    ---------
    fluxes: numpy ndarray
        pixel intensities
    ivars: numpy ndarray
        inverse variances, parallel to fluxes
    cont: numpy ndarray
        the continuum
    ranges: list or np ndarray
        the chunks that the spectrum should be split into

    Returns
    -------
    norm_fluxes: numpy ndarray
        normalized pixel intensities
    norm_ivars: numpy ndarray
        rescaled inverse variances
    """
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm(fluxes[:,start:stop],
                           ivars[:,start:stop],
                           cont[:,start:stop])
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    for jj in range(nstars):
        bad = (norm_ivars[jj,:] == 0.)
        norm_fluxes[jj,:][bad] = 0.
        norm_ivars[jj,:][bad] = SMALL**2
    return norm_fluxes, norm_ivars
