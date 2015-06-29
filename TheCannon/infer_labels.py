from __future__ import (absolute_import, division, print_function, unicode_literals)

from scipy import optimize as opt
import numpy as np

LARGE = 200.
SMALL = 1. / LARGE

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


def _infer_labels(model, dataset):
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
    chisqs = model.chisqs
    pivots = model.pivots
    nlabels = dataset.tr_label.shape[1] 
    fluxes = dataset.test_flux
    ivars = dataset.test_ivar
    nstars = fluxes.shape[0]
    labels_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, coeffs_all.shape[1] - 1,
                               coeffs_all.shape[1]-1.))
    errs_all = np.zeros((nstars, nlabels))

    for jj in range(nstars):
        print(jj)
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        flux_piv = flux - coeffs_all[:,0] * 1.  # pivot around the leading term
        sig = np.sqrt(1./ivar + scatters**2)
        coeffs = np.delete(coeffs_all, 0, axis=1)  # take pivot into account
        try:
            labels, covs = opt.curve_fit(_func, coeffs, flux_piv,
                                         p0=np.repeat(1, nlabels),
                                         sigma=sig, absolute_sigma=True)
        except TypeError:  # old scipy version
            labels, covs = opt.curve_fit(_func, coeffs, flux_piv,
                                         p0=np.repeat(1, nlabels), sigma=sig)
            # rescale covariance matrix
            chi = (flux_piv-_func(coeffs, *labels)) / sig
            chi2 = (chi**2).sum()
            # FIXME: dof does not seem to be right to me (MF)
            dof = len(flux_piv) - nlabels
            factor = (chi2 / dof)
            covs /= factor
        labels = labels + pivots
        labels_all[jj,:] = labels
        errs_all[jj,:] = covs.diagonal()
    dataset.set_test_label_vals(labels_all)
    return errs_all
