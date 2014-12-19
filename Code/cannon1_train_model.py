# This is Step 1 of The Cannon
from stars import Stars
import numpy as np
import math
import pylab

nstars = 0 # number of training stars. we don't deal with test stars here.
npixels = 0
nlabels = 0

def do_one_regression_at_fixed_scatter(spectra, x, scatter):
    """
    Params
    ------
    spectra: ndarray, [nstars, 3]
        "3" corresponds to wavelength, flux, flux_err

    x: ndarray, [nstars, nlabels]

    scatter:

    Returns
    ------
    coeff: ndarray
        coefficients of the fit

    xTCinvx: ndarray
        inverse covariance matrix for fit coefficients

    chi: float
        chi-squared at best fit

    logdet_Cinv: float
        inverse of the log determinant of the cov matrix
    """
    Cinv = 1. / (spectra[:, 2] ** 2 + scatter ** 2)  
    xTCinvx = np.dot(x.T, Cinv[:, None] * x) # craziness b/c Cinv isnt a matrix
    fluxes = spectra[:, 1] 
    xTCinvf = np.dot(x.T, Cinv * fluxes)
    try:
        coeff = np.linalg.solve(xTCinvx, xTCinvf) # this is the model!
    except np.linalg.linalg.LinAlgError:
        print "np.linalg.linalg.LinAlgError in do_one_regression_at_fixed_scatter"
        print MTCinvM, MTCinvx, spectra[:,0], spectra[:,1], spectra[:,2]
        print fluxes
    assert np.all(np.isfinite(coeff))
    chi = np.sqrt(Cinv) * (fluxes - np.dot(x, coeff))
    logdet_Cinv = np.sum(np.log(Cinv))
    return (coeff, xTCinvx, chi, logdet_Cinv)

def do_one_regression(spectra, x):
    """do_one_regression
    Optimizes to solve for coeffs (the model) and determines the scatter of the model at a single wavelength for all stars.
    (The scatter - the deviation between obs. spectrum and model - is presumed to be wavelength-dependent) 
    """
    ln_scatter_vals = np.arange(np.log(0.0001), 0., 0.5) # where does this come from?!
    chis_eval = np.zeros_like(ln_scatter_vals)
    for jj, ln_scatter_val in enumerate(ln_scatter_vals):
        coeff, xTCinvx, chi, logdet_Cinv = do_one_regression_at_fixed_scatter(spectra, x, scatter = np.exp(ln_scatter_val))
        chis_eval[jj] = np.sum(chi * chi) - logdet_Cinv
    # What do the below two cases *mean*?
    if np.any(np.isnan(chis_eval)):
        best_scatter = np.exp(ln_scatter_vals[-1]) # don't really understand this
        return do_one_regression_at_fixed_scatter(spectra, x, scatter = best_scatter) + (best_scatter, )
    lowest = np.argmin(chis_eval) # the best-fit scatter value?
    if lowest == 0 or lowest == len(ln_scatter_vals) + 1: # if it didn't really find it
        best_scatter = np.exp(ln_scatter_vals[lowest])
        return do_one_regression_at_fixed_scatter(spectra, x, scatter = best_scatter) + (best_scatter, )
    ln_scatter_vals_short = ln_scatter_vals[np.array([lowest-1, lowest, lowest+1])]
    chis_eval_short = chis_eval[np.array([lowest-1, lowest, lowest+1])]
    z = np.polyfit(ln_scatter_vals_short, chis_eval_short, 2)
    f = np.poly1d(z)
    fit_pder = np.polyder(z)
    fit_pder2 = pylab.polyder(fit_pder)
    best_scatter = np.exp(np.roots(fit_pder)[0])
    return do_one_regression_at_fixed_scatter(spectra, x, scatter = best_scatter) + (best_scatter, )

def train_model(training_set):
    """
    This determines the coefficients from the training data
    """
    label_names = training_set.get_label_names()
    label_values = training_set.get_label_values() #(nstars, nlabels)
    global nlabels
    nlabels = len(label_names)
    spectra = training_set.get_spectra() #(nstars, npixels, 3)
    global nstars 
    nstars = spectra.shape[0]
    global npixels 
    npixels = spectra.shape[1]

    # Each star has a LABEL VECTOR: x = [1, l1-L1, l2-L2, etc] where Lk is the mean value of that label across all the stars. Each element is called an offset, and these mean values are called pivots. We assume in this version of the code that our model is quadratic in labels, but in the future we want to generalize this and even allow each pixel to have its own functional form. 
    pivots = np.mean(label_values, axis=0)
    ones = np.ones((nstars, 1))
    linear_offsets = label_values - pivots
    quadratic_offsets = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (label_values - pivots)])
    x = np.hstack((ones, linear_offsets, quadratic_offsets))
    x_full = np.array([x,]*npixels) # (npixels, nstars, 15)

    # Perform REGRESSIONS

    spectra = spectra.swapaxes(0,1) # for consistency with x_full
    blob = map(do_one_regression, spectra, x_full) # one regression per pixel
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    chisqs = np.array([np.dot(b[2],b[2]) - b[3] for b in blob]) # holy crap be careful
    scatters = np.array([b[4] for b in blob])

    model = coeffs, covs, scatters, chis, chisqs, pivots
    return model
