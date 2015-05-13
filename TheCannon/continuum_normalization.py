import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize as opt

""" Performs continuum normalization on Cannon input spectra. """

LARGE = 200.
SMALL = 1. / LARGE

def partial_func(func, *args, **kwargs):
    def wrap(x, *p):
        return func(x, p, **kwargs)
    return wrap


def cont_func(x, p, L, y):
    """ Return the fitting function evaluated at input x for the continuum.
    The fitting function is a sinusoid, sum of sines and cosines

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


def fit_cont(fluxes, ivars, contmask, deg, ffunc):
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
    print("order: %s" %deg)
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
            pcont_func = partial_func(cont_func, L=L, y=flux)
            popt, pcov = opt.curve_fit(pcont_func, x, y, p0=p0, 
                                       sigma=1./np.sqrt(yivar))
        elif ffunc=="chebyshev":
            fit = np.polynomial.chebyshev.Chebyshev.fit(x=x,y=y,w=yivar,deg=deg)
        for element in pix:
            if ffunc=="sinusoid":
                cont[jj,element] = cont_func(element, popt, L=L, y=flux)
            elif ffunc=="chebyshev":
                cont[element] = fit(element)
    return cont


def fit_cont_regions(fluxes, ivars, contmask, deg, ranges, ffunc):
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
    print("taking spectra in %s regions" %len(ranges))
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    cont = np.zeros(fluxes.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        if ffunc=="chebyshev":
            output = fit_cont(fluxes[:,start:stop],
                              ivars[:,start:stop],
                              contmask[start:stop], deg=deg, ffunc="chebyshev")
        elif ffunc=="sinusoid":
            output = fit_cont(fluxes[:,start:stop],
                              ivars[:,start:stop],
                              contmask[start:stop], deg=deg, ffunc="sinusoid")
        cont[:,start:stop] = output
    return cont


def cont_norm(fluxes, ivars, cont):
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


def weighted_median(values, weights, quantile):
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


def cont_norm_q(wl, fluxes, ivars, q, delta_lambda):
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
            cont[jj,ll] = weighted_median(flux_cut, ivar_cut, q)
    for jj in range(nstars):
        norm_fluxes[jj,:] = fluxes[jj,:]/cont[jj,:]
        norm_ivars[jj,:] = cont[jj,:]**2 * ivars[jj,:]
    return norm_fluxes, norm_ivars


def cont_norm_regions(fluxes, ivars, cont, ranges):
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
    print("taking spectra in %s regions" %len(ranges))
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = cont_norm(fluxes[:,start:stop],
                           ivars[:,start:stop],
                           cont[:,start:stop])
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    for jj in range(nstars):
        bad = (norm_ivars[jj,:] == 0.)
        norm_fluxes[jj,:][bad] = 0.
        norm_ivars[jj,:][bad] = SMALL**2
    return norm_fluxes, norm_ivars
