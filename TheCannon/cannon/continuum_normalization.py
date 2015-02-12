import numpy as np
import scipy.optimize as opt

""" Performs continuum normalization on Cannon input spectra. """

def cont_func(x, *p):
    """ Return the fitting function for the continuum.

    Parameters
    ----------
    x: ndarray
        x values of data to fit
    p: ndarray
        function coefficients

    Returns
    -------
    func: float
        function evaluated for the input x
    """
    N = int(len(p)/2)
    n = np.linspace(0, N, N+1, dtype=int)
    L = max(x)-min(x)
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
        y = flux[contmask]
        x = np.arange(0, npixels)[contmask]
        yivar = ivar[contmask]
        p0 = np.ones(deg*2) # one for cos, one for sin
        popt, pcov = opt.curve_fit(cont_func, x, y, p0=p0, sigma=1./np.sqrt(yivar))
        cont = cont_func(x, popt)
        norm_fluxes[jj,:] = flux/cont
        norm_ivars[jj,:] = cont**2 * ivar

    return norm_fluxes, norm_ivars

def cont_norm_segments(fluxes, ivars, contmask, deg=3):
    """ Continuum-normalize a list of continuous spectrum segments.

    Loops through segments and calls cont_norm
    """
    nseg = len(fluxes)
    norm_fluxes = np.zeros((2,)+fluxes.shape)
    norm_ivars = norm_fluxes.shape
    for seg in nseg:
        norm_flux, norm_ivar = cont_norm(fluxes[seg], ivars[seg], 
                                         contmask, deg=deg)
        norm_fluxes[seg,:,:] = norm_flux
        norm_ivars[seg,:,:] = norm_ivar
    return norm_fluxes, norm_ivars
