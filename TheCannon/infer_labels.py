from __future__ import (absolute_import, division, print_function, unicode_literals)

from scipy import optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from TheCannon import train_model

def _get_lvec(labels):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels

    Parameters
    ----------
    labels: numpy ndarray
        pivoted label values for one star

    Returns
    -------
    lvec: numpy ndarray
        label vector
    """
    nlabels = len(labels)
    # specialized to second-order model
    linear_terms = labels 
    quadratic_terms = np.outer(linear_terms, 
                               linear_terms)[np.triu_indices(nlabels)]
    lvec = np.hstack((linear_terms, quadratic_terms))
    return lvec


def _func(coeffs, *labels):
    """ Takes the dot product of coefficients vec & labels vector 
    
    Parameters
    ----------
    coeffs: numpy ndarray
        the coefficients on each element of the label vector

    *labels: numpy ndarray
        label vector

    Returns
    -------
    dot product of coeffs vec and labels vec
    """
    lvec = _get_lvec(list(labels))
    return np.dot(coeffs, lvec)


def _infer_labels(model, dataset, starting_guess=None):
    """
    Uses the model to solve for labels of the test set.

    Parameters
    ----------
    model: tuple
        Coeffs_all, covs, scatters, chis, chisqs, pivots

    dataset: Dataset
        Dataset that needs label inference

    Returns
    -------
    errs_all:
        Covariance matrix of the fit
    """
    print("Inferring Labels")
    coeffs_all = model.coeffs
    scatters = model.scatters
    #chisqs = model.chisqs
    nlabels = len(dataset.get_plotting_labels())
    fluxes = dataset.test_flux
    ivars = dataset.test_ivar
    nstars = fluxes.shape[0]
    labels_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, coeffs_all.shape[1] - 1,
                               coeffs_all.shape[1] - 1))
    errs_all = np.zeros((nstars, nlabels))
    chisq_all = np.zeros(nstars)
    scales = model.scales

    if starting_guess is None:
        starting_guess = np.ones(nlabels)

    # print("starting guess: %s" %starting_guess)
    for jj in range(nstars):
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        

        # where the ivar == 0, set the normalized flux to 1 and the sigma to 100
        bad = ivar == 0
        flux[bad] = 1.0
        sigma = np.ones(ivar.shape) * 100.0
        sigma[~bad] = np.sqrt(1.0 / ivar[~bad])

        flux_piv = flux - coeffs_all[:,0] * 1.  # pivot around the leading term
        errbar = np.sqrt(sigma**2 + scatters**2)
        coeffs = np.delete(coeffs_all, 0, axis=1)  # take pivot into account
        
        try:
            labels, covs = opt.curve_fit(_func, coeffs, flux_piv,
                                         p0 = starting_guess,
                                         sigma=errbar, absolute_sigma=True)
        except RuntimeError:
            print("Error - curve_fit failed")
            labels = np.zeros(starting_guess.shape)-9999.
            covs = np.zeros((len(starting_guess),len(starting_guess)))-9999.
        chi2 = (flux_piv-_func(coeffs, *labels))**2 * ivar / (1 + ivar * scatters**2)
        chisq_all[jj] = sum(chi2)
        labels_all[jj,:] = model.scales * labels + model.pivots
        errs_all[jj,:] = np.sqrt(covs.diagonal())

    dataset.set_test_label_vals(labels_all)
    return errs_all, chisq_all
    
    
    
    
    
    
