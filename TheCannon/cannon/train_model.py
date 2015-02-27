"""This runs Step 1 of The Cannon:
    uses the training set to solve for the best-fit model."""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
from .helpers.compatibility import range, map
from .helpers.triangle import corner

def do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec, scatter):
    """
    Parameters
    ----------
    lams: numpy ndarray, shape npixels
        the common wavelength array

    fluxes: numpy ndarray, shape (nstars, npixels)
        flux values for all pixels of all stars
        
    ivars: numpy ndarray, shape (nstars, npixels)
        inverse variance values for all pixels of all stars

    lvec: numpy ndarray
        the label vector

    scatter: numpy ndarray, shape npixels
        fitted scatter values

    Returns
    ------
    coeff: ndarray
        coefficients of the fit

    lTCinvl: ndarray
        inverse covariance matrix for fit coefficients
    
    chi: float
        chi-squared at best fit
    
    logdet_Cinv: float
        inverse of the log determinant of the cov matrix
    """
    sig2 = 100**2*np.ones(len(ivars))
    mask = ivars != 0
    sig2[mask] = 1. / ivars[mask]
    Cinv = 1. / (sig2 + scatter**2)
    lTCinvl = np.dot(lvec.T, Cinv[:, None] * lvec)
    lTCinvf = np.dot(lvec.T, Cinv * fluxes)
    try:
        coeff = np.linalg.solve(lTCinvl, lTCinvf)
    except np.linalg.linalg.LinAlgError:
        print("np.linalg.linalg.LinAlgError, do_one_regression_at_fixed_scatter")
        print(lTCinvl, lTCinvf, lams, fluxes)
    if not np.all(np.isfinite(coeff)):
        raise RuntimeError('something is wrong with the coefficients')
    chi = np.sqrt(Cinv) * (fluxes - np.dot(lvec, coeff))
    logdet_Cinv = np.sum(np.log(Cinv))
    return (coeff, lTCinvl, chi, logdet_Cinv)


def do_one_regression(lams, fluxes, ivars, lvec):
    """
    Optimizes to find the scatter associated with the best-fit model.

    This scatter is the deviation between the observed spectrum and the model.
    It is wavelength-independent, so we perform this at a single wavelength.

    Input
    -----
    lams: numpy ndarray, shape (npixels)
        the common wavelength array

    fluxes: numpy ndarray, shape (nstars)

    ivars: numpy ndarray, shape (nstars)

    lvec = numpy ndarray 
        the label vector

    Output
    -----
    output of do_one_regression_at_fixed_scatter
    """
    ln_scatter_vals = np.arange(np.log(0.0001), 0., 0.5)
    # minimize over the range of scatter possibilities
    chis_eval = np.zeros_like(ln_scatter_vals)
    for jj, ln_scatter_val in enumerate(ln_scatter_vals):
        coeff, lTCinvl, chi, logdet_Cinv = \
            do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec,
                                               scatter=np.exp(ln_scatter_val))
        chis_eval[jj] = np.sum(chi*chi) - logdet_Cinv
    if np.any(np.isnan(chis_eval)):
        best_scatter = np.exp(ln_scatter_vals[-1])
        _r = do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec,
                                                scatter=best_scatter)
        return _r + (best_scatter, )
    lowest = np.argmin(chis_eval)
    if (lowest == 0) or (lowest == len(ln_scatter_vals) + 1):
        best_scatter = np.exp(ln_scatter_vals[lowest])
        _r = do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec,
                                                scatter=best_scatter)
        return _r + (best_scatter, )
    ln_scatter_vals_short = ln_scatter_vals[np.array(
        [lowest-1, lowest, lowest+1])]
    chis_eval_short = chis_eval[np.array([lowest-1, lowest, lowest+1])]
    z = np.polyfit(ln_scatter_vals_short, chis_eval_short, 2)
    fit_pder = np.polyder(z)
    best_scatter = np.exp(np.roots(fit_pder)[0])
    _r = do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec,
                                            scatter=best_scatter)
    return _r + (best_scatter, )

def get_lvec(label_vals, pivots):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels

    Parameters
    ----------
    label_vals: numpy ndarray, shape (nstars, nlabels)
    pivots: array corresponding to the mean of the label_vals

    Returns
    -------
    lvec_full: numpy ndarray
        label vector
    """
    nlabels = label_vals.shape[1]
    nstars = label_vals.shape[0]
    # specialized to second-order model
    linear_offsets = label_vals - pivots
    quadratic_offsets = np.array([np.outer(m, m)[np.triu_indices(nlabels)]
                                  for m in (linear_offsets)])
    ones = np.ones((nstars, 1))
    lvec = np.hstack((ones, linear_offsets, quadratic_offsets))
    return lvec

def train_model(dataset):
    """
    This determines the coefficients of the model using the training data

    Parameters
    ----------
    reference_set: Dataset

    Returns
    -------
    model, which consists of:
   
    coeffs: ndarray, (npixels, nstars, 15)
        the model coefficients

    covs: ndarray
        the covariance matrix

    scatter:
        scatter values

    all_chisqs:

    pivots:
        the pivot values

    lvec_full:
        the label vector
    """
    print("Training model...")
    label_names = dataset.label_names
    label_vals = dataset.tr_label_vals
    lams = dataset.wl
    npixels = len(lams)
    fluxes = dataset.tr_fluxes
    ivars = dataset.tr_ivars
    pivots = np.mean(label_vals, axis=0)
    lvec = get_lvec(label_vals, pivots)
    lvec_full = np.array([lvec,] * npixels)

    # Perform REGRESSIONS
    fluxes = fluxes.swapaxes(0,1)  # for consistency with lvec_full
    ivars = ivars.swapaxes(0,1)
    # one per pix
    blob = list(map(do_one_regression, lams, fluxes, ivars, lvec_full))
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    scatters = np.array([b[4] for b in blob])

    # Calc chi sq
    all_chisqs = chis*chis
    model = coeffs, covs, scatters, all_chisqs, pivots, lvec_full
    print("Done training model")
    return model


def split_array(array, num):
    avg = len(array) / float(num)
    out = []
    last = 0.0
    while last < len(array):
        out.append(array[int(last):int(last+avg)])
        last += avg
    return out
