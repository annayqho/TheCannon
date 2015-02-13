import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import scipy.optimize as opt

""" Performs continuum normalization on Cannon input spectra. """

# Thank you Morgan for this...
def partial_func(func, *args, **kwargs):
    
    def wrap(x, *p):
        return func(x, p, **kwargs)
    return wrap

def cont_func(x, p, L):
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
    func = 0.
    for n in range(0, N):
        func += p[2*n]*np.sin(k[n]*x)+p[2*n+1]*np.cos(k[n]*x)
    return func

def cont_norm(fluxes, ivars, contmask, deg=3):
    """ Continuum-normalize a continuous segment of spectra.

    Fit a function of sines and cosines and divide the spectra by it

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
    npixels = fluxes.shape[1]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    
    for jj in range(nstars):
        # Fit continuum to cont pixels
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        pix = np.arange(0, npixels)
        y = flux[contmask]
        x = pix[contmask]
        yivar = ivar[contmask]
        p0 = np.ones(deg*2) # one for cos, one for sin
        L = max(x)-min(x)
        pcont_func = partial_func(cont_func, L=L)
        popt, pcov = opt.curve_fit(pcont_func, x, y, p0=p0, 
                                   sigma=1./np.sqrt(yivar))
        cont = np.zeros(len(pix))
        for element in pix:
            cont[element] = cont_func(element, popt, L=L)
        #plt.scatter(pix, cont)
        #plt.show()
        norm_fluxes[jj,:] = flux/cont
        norm_ivars[jj,:] = cont**2 * ivar

    return norm_fluxes, norm_ivars

def cont_norm_regions(fluxes, ivars, contmask, ranges, deg=3):
    print("taking spectra in %s regions" %len(ranges))
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = cont_norm(fluxes[:,start:stop],
                           ivars[:,start:stop],
                           contmask[start:stop])
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    return norm_fluxes, norm_ivars

