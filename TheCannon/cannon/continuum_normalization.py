import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize as opt

""" Performs continuum normalization on Cannon input spectra. """

LARGE = 200.
SMALL = 1. / LARGE

# Thank you Morgan for this...
def partial_func(func, *args, **kwargs):
    def wrap(x, *p):
        return func(x, p, **kwargs)
    return wrap

def cont_func(x, p, L, y):
    """ Return the fitting function for the continuum.

    Parameters
    ----------
    x: float or np.array
    p: ndarray
        function coefficients. first element L is not fitted for.

    Returns
    -------
    func: float
        function evaluated for the input x
    """
    N = int(len(p)/2)
    n = np.linspace(0, N, N+1, dtype=int)
    k = n*np.pi/L
    baseline = 0 #if you were fitting a flat spectrum...
    #baseline = y[x]
    func = 0
    for n in range(0, N):
        func += p[2*n]*np.sin(k[n]*x)+p[2*n+1]*np.cos(k[n]*x)
    return func
    #return baseline+func


def fit_cont(fluxes, ivars, contmask, deg):
    """ Fit a continuum to a continuous segment of spectra.

    Fit a function of sines and cosines with specified degree.
    """
    print("order: %s" %deg)
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    cont = np.zeros(fluxes.shape)
    
    for jj in range(nstars):
        print(jj)
        # Fit continuum to cont pixels
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        pix = np.arange(0, npixels)
        y = flux[contmask]
        x = pix[contmask]
        yivar = ivar[contmask]
        yivar[yivar == 0] = SMALL**2 # for nont cont norm spectra 
        p0 = np.ones(deg*2) # one for cos, one for sin
        L = max(x)-min(x)
        pcont_func = partial_func(cont_func, L=L, y=flux)
        # It should not be possible to have a cont pix that's also
        # a bad pixel, because bad means flux_err==0, and flux_err==0
        # should also correspond to flux==0. If flux==0 at this pixel,
        # it should throw off the median(flux) and ivar(flux) cuts.
        # bad = yivar == 0 #| yivar == SMALL 
        # yivar = np.ma.array(yivar, mask=bad)
        
        # in the sine/cosine version:
        #popt, pcov = opt.curve_fit(pcont_func, x, y, p0=p0, 
        #                           sigma=1./np.sqrt(yivar))
        fit = np.polynomial.chebyshev.Chebyshev.fit(x=x,y=y,w=yivar,deg=3)

        for element in pix:
            # sine/cosine version:
            # cont[jj,element] = cont_func(element, popt, L=L, y=flux)
            cont[element] = fit(element)

    return cont

def fit_cont_regions(fluxes, ivars, contmask, deg, ranges):
    print("taking spectra in %s regions" %len(ranges))
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    cont = np.zeros(fluxes.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = fit_cont(fluxes[:,start:stop],
                          ivars[:,start:stop],
                          contmask[start:stop], deg=deg)
        cont[:,start:stop] = output
    return cont

def cont_norm(fluxes, ivars, cont):
    """ Continuum-normalize a continuous segment of spectra.

    Fit has already been performed.

    Parameters
    ----------
    fluxes: numpy ndarray 
        pixel intensities
    ivars: numpy ndarray 
        inverse variances, parallel to fluxes
    contmask: boolean mask
        True indicates that pixel is continuum
    deg: (optional) int
        degree of fit, corresponds to # of sines or # of cosines

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
    sindx = np.argsort(values)
    cvalues = 1. * np.cumsum(weights[sindx])
    cvalues = cvalues / cvalues[-1]
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    indx = foo[0]
    return values[indx]

def cont_norm_q(wl, fluxes, ivars, q=0.90, delta_lambda=50):
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    cont = np.zeros(fluxes.shape)
    nstars = fluxes.shape[0]
    for jj in range(nstars):
        print "cont_norm_q(): working on star", jj
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
        bad = (ivars[jj,:] == SMALL**2) 
        norm_fluxes[jj,:][bad] = 0.
        norm_ivars[jj,:][bad] = SMALL**2
    return norm_fluxes, norm_ivars


def cont_norm_regions(fluxes, ivars, contmask, ranges, deg=3):
    print("taking spectra in %s regions" %len(ranges))
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = cont_norm(fluxes[:,start:stop],
                           ivars[:,start:stop],
                           contmask[start:stop], deg=deg)
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    for jj in range(nstars):
        bad = (norm_ivars[jj,:] == 0.)
        #bad = (norm_ivars[jj,:] == SMALL**2)
        norm_fluxes[jj,:][bad] = 0.
        norm_ivars[jj,:][bad] = SMALL**2
    return norm_fluxes, norm_ivars

