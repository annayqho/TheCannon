import numpy as np
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.optimize as opt
# from joblib import Parallel, delayed

SMALL = 1.0/200

def _partial_func(func, *args, **kwargs):
    def wrap(x, *p):
        return func(x, p, **kwargs)
    return wrap


def gaussian_weight_matrix(wl, L):
    """ Matrix of Gaussian weights 

    Parameters
    ----------
    wl: numpy ndarray
        pixel wavelength values
    L: float
        width of Gaussian

    Return
    ------
    Weight matrix
    """
    return np.exp(-0.5*(wl[:,None]-wl[None,:])**2/L**2)


def _sinusoid(x, p, L, y):
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
    if cvalues[-1] == 0: # means all the values are 0
        return values[0]
    cvalues = cvalues / cvalues[-1] # div by largest value
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    indx = foo[0]
    return values[indx]


def _find_cont_gaussian_smooth(wl, fluxes, ivars, w):
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
    print("Finding the continuum")
    bot = np.dot(ivars, w.T)
    top = np.dot(fluxes*ivars, w.T)
    bad = bot == 0
    cont = np.zeros(top.shape)
    cont[~bad] = top[~bad] / bot[~bad]
    return cont


def _cont_norm_gaussian_smooth(dataset, L):
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
    print("Gaussian smoothing the entire dataset...")
    w = gaussian_weight_matrix(dataset.wl, L)

    print("Gaussian smoothing the training set")
    cont = _find_cont_gaussian_smooth(
            dataset.wl, dataset.tr_flux, dataset.tr_ivar, w)
    norm_tr_flux, norm_tr_ivar = _cont_norm(
            dataset.tr_flux, dataset.tr_ivar, cont)
    print("Gaussian smoothing the test set")
    cont = _find_cont_gaussian_smooth(
            dataset.wl, dataset.test_flux, dataset.test_ivar, w)
    norm_test_flux, norm_test_ivar = _cont_norm(
            dataset.test_flux, dataset.test_ivar, cont)
    return norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar 


def _find_cont_fitfunc(fluxes, ivars, contmask, deg, ffunc, n_proc=1):
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

    if n_proc == 1:
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
                pcont_func = _partial_func(_sinusoid, L=L, y=flux)
                popt, pcov = opt.curve_fit(pcont_func, x, y, p0=p0,
                                           sigma=1./np.sqrt(yivar))
            elif ffunc=="chebyshev":
                fit = np.polynomial.chebyshev.Chebyshev.fit(x=x,y=y,w=yivar,deg=deg)
            for element in pix:
                if ffunc=="sinusoid":
                    cont[jj,element] = _sinusoid(element, popt, L=L, y=flux)
                elif ffunc=="chebyshev":
                    cont[jj,element] = fit(element)
    else:
        # start mp.Pool
        pool = mp.Pool(processes=n_proc)
        mp_results = []
        for i in xrange(nstars):
            mp_results.append(pool.apply_async(\
                _find_cont_fitfunc,
                (fluxes[i, :].reshape((1, -1)),
                 ivars[i, :].reshape((1, -1)),
                 contmask[:]),
                {'deg':deg, 'ffunc':ffunc}))
        # close mp.Pool
        pool.close()
        pool.join()

        cont = np.array([mp_results[i].get().flatten() for i in xrange(nstars)])

    return cont


def _find_cont_fitfunc_regions(fluxes, ivars, contmask, deg, ranges, ffunc,
                               n_proc=1):
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
            output = _find_cont_fitfunc(fluxes[:,start:stop],
                                        ivars[:,start:stop],
                                        contmask[start:stop],
                                        deg=deg, ffunc="chebyshev",
                                        n_proc=n_proc)
        elif ffunc=="sinusoid":
            output = _find_cont_fitfunc(fluxes[:,start:stop],
                                        ivars[:,start:stop],
                                        contmask[start:stop],
                                        deg=deg, ffunc="sinusoid",
                                        n_proc=n_proc)
        cont[:, start:stop] = output

    return cont


def _find_cont_running_quantile(wl, fluxes, ivars, q, delta_lambda,
                                verbose=False):
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
    cont = np.zeros(fluxes.shape)
    nstars = fluxes.shape[0]
    for jj in range(nstars):
        if verbose:
            print("cont_norm_q(): working on star [%s/%s]..." % (jj+1, nstars))
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        for ll, lam in enumerate(wl):
            indx = (np.where(abs(wl-lam) < delta_lambda))[0]
            flux_cut = flux[indx]
            ivar_cut = ivar[indx]
            cont[jj, ll] = _weighted_median(flux_cut, ivar_cut, q)
    return cont


####################################################
# structure of the continuum normalization process
#
# continuum_normalize_training_q()
# |- _cont_norm_running_quantile()
#    |- _find_cont_running_quantile()
#    |- _find_cont_running_quantile_mp()
# |- _cont_norm_running_quantile_regions
#    |- _cont_norm_running_quantile
#    |- _cont_norm_running_quantile_mp()
#
# the basic running_quantile method:
# _find_cont_running_quantile()
####################################################


def _cont_norm_running_quantile(wl, fluxes, ivars, q, delta_lambda, verbose=True):
    cont = _find_cont_running_quantile(wl, fluxes, ivars, q, delta_lambda, verbose=verbose)
    norm_fluxes = np.ones(fluxes.shape)
    # norm_ivars = np.zeros(ivars.shape)
    norm_fluxes[cont!=0] = fluxes[cont!=0] / cont[cont!=0]
    norm_ivars = cont**2 * ivars
    return norm_fluxes, norm_ivars

def _cont_norm_running_quantile_mp(wl, fluxes, ivars, q, delta_lambda,
                                   n_proc=2, verbose=False):
    """
    The same as _cont_norm_running_quantile() above,
    but using multi-processing.

    Bo Zhang (NAOC)
    """
    nStar = fluxes.shape[0]

    # start mp.Pool
    mp_results = []
    pool = mp.Pool(processes=n_proc)
    for i in xrange(nStar):
        mp_results.append(pool.apply_async(\
            _find_cont_running_quantile,
            (wl, fluxes[i, :].reshape((1, -1)), ivars[i, :].reshape((1, -1)),
             q, delta_lambda), {'verbose': False}))
        if verbose:
            print('@Bo Zhang: continuum normalizing star [%d/%d] ...'\
                  % (i + 1, nStar))
    # close mp.Pool
    pool.close()
    pool.join()

    # reshape results --> cont
    cont = np.zeros_like(fluxes)
    for i in xrange(nStar):
        cont[i, :] = mp_results[i].get() #.flatten()
    norm_fluxes = np.ones(fluxes.shape)
    norm_fluxes[cont!=0] = fluxes[cont!=0] / cont[cont!=0]
    norm_ivars = cont**2 * ivars

    print('@Bo Zhang: continuum normalization finished!')
    return norm_fluxes, norm_ivars

    # if use joblib
    # mp_results = Parallel(n_jobs=n_proc)\
    #     (delayed(_find_cont_running_quantile)\
    #          (wl, fluxes[i, :].reshape((1, -1)), ivars[i, :].reshape((1, -1)), q, delta_lambda, False, True) for i in range(nStar))
    # return mp_results


def _cont_norm_running_quantile_regions(wl, fluxes, ivars, q, delta_lambda,
                                        ranges, verbose=True):
    """ Perform continuum normalization using running quantile, for spectrum
    that comes in chunks
    """
    print("contnorm.py: continuum norm using running quantile")
    print("Taking spectra in %s chunks" % len(ranges))
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm_running_quantile(
                wl[start:stop], fluxes[:,start:stop],
                ivars[:,start:stop], q, delta_lambda)
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    return norm_fluxes, norm_ivars


def _cont_norm_running_quantile_regions_mp(wl, fluxes, ivars, q, delta_lambda,
                                           ranges, n_proc=2, verbose=False):
    """
    Perform continuum normalization using running quantile, for spectrum
    that comes in chunks.

    The same as _cont_norm_running_quantile_regions(),
    but using multi-processing.

    Bo Zhang (NAOC)
    """
    print("contnorm.py: continuum norm using running quantile")
    print("Taking spectra in %s chunks" % len(ranges))
    # nstars = fluxes.shape[0]
    nchunks = len(ranges)
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for i in xrange(nchunks):
        chunk = ranges[i, :]
        start = chunk[0]
        stop = chunk[1]
        if verbose:
            print('@Bo Zhang: Going to normalize Chunk [%d/%d], pixel:[%d, %d] ...'
                  % (i+1, nchunks, start, stop))
        output = _cont_norm_running_quantile_mp(
            wl[start:stop], fluxes[:, start:stop],
            ivars[:, start:stop], q, delta_lambda,
            n_proc=n_proc, verbose=verbose)
        norm_fluxes[:, start:stop] = output[0]
        norm_ivars[:, start:stop] = output[1]
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
    npixels = fluxes.shape[1]
    norm_fluxes = np.ones(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    bad = cont == 0.
    norm_fluxes = np.ones(fluxes.shape)
    norm_fluxes[~bad] = fluxes[~bad] / cont[~bad]
    norm_ivars = cont**2 * ivars
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
        norm_fluxes[jj,:][bad] = 1.
    return norm_fluxes, norm_ivars
