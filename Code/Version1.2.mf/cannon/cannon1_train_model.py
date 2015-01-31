"""This runs Step 1 of The Cannon:
    uses the training set to solve for the best-fit model."""

from __future__ import (absolute_import, division, print_function, unicode_literals)

# Currently, this is a bit sketchy...there were sections of the original code
# that I didn't understand. have e-mailed MKN, will update.

# imported but not used
# from dataset import Dataset
# import math
# from matplotlib import rc
# import os
# import pylab

import numpy as np
import matplotlib.pyplot as plt
from .helpers.compatibility import range, map


def do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec, scatter):
    """
    Params
    ------
    lams: ndarray, [npixels]
    spectra: ndarray, [nstars, 3]
    lvec=label vector: ndarray, [nstars, 10]
    scatter: ndarray, [nstars]

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
        # FIXME: undefined name 'lTCinvx'
        # print(lTCinvx, lTCinvf, lams, fluxes)
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
    lams: ndarray, [npixels]
    spectra: ndarray, [nstars, 2]
    x = the label vector, ndarray, [nstars, 10]

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
    # assigned but never used
    # f = np.poly1d(z)
    fit_pder = np.polyder(z)
    # assigned but never used
    # fit_pder2 = pylab.polyder(fit_pder)
    best_scatter = np.exp(np.roots(fit_pder)[0])
    _r = do_one_regression_at_fixed_scatter(lams, fluxes, ivars, lvec,
                                            scatter=best_scatter)
    return _r + (best_scatter, )


def train_model(reference_set):
    """
    This determines the coefficients of the model using the training data

    Input: the reference_set, a Dataset object (see dataset.py)
    Returns: the model, which consists of...
    -------
    coefficients: ndarray, (npixels, nstars, 15)
    the covariance matrix
    scatter values
    red chi squareds
    the pivot values
    the label vector
    """
    print("Training model...")
    label_names = reference_set.label_names
    label_vals = reference_set.label_vals
    nlabels = len(label_names)
    lams = reference_set.lams
    fluxes = reference_set.fluxes
    ivars = reference_set.ivars
    nstars = fluxes.shape[0]
    npixels = len(lams)

    # Establish label vector
    pivots = np.mean(label_vals, axis=0)
    ones = np.ones((nstars, 1))
    linear_offsets = label_vals - pivots
    quadratic_offsets = np.array([np.outer(m, m)[np.triu_indices(nlabels)]
                                  for m in (label_vals - pivots)])
    lvec = np.hstack((ones, linear_offsets, quadratic_offsets))
    lvec_full = np.array([lvec,]*npixels)

    # Perform REGRESSIONS
    fluxes = fluxes.swapaxes(0,1)  # for consistency with x_full
    ivars = ivars.swapaxes(0,1)
    # one per pix
    blob = list(map(do_one_regression, lams, fluxes, ivars, lvec_full))
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    scatters = np.array([b[4] for b in blob])

    # Calc chi sq
    all_chisqs = chis*chis
    # chisqs = np.sum(all_chisqs, axis=0) # now we have one per star
    model = coeffs, covs, scatters, all_chisqs, pivots, lvec_full
    print("Done training model")
    return model


def model_diagnostics(reference_set, model):
    """Run a set of diagnostics on the model.

    Plot the 0th order coefficients as the baseline spectrum.
    Overplot the continuum pixels.

    Plot each label's leading coefficient as a function of wavelength.
    Color-code by label.

    Histogram of the chi squareds of the fits.
    Dotted line corresponding to DOF = npixels - nlabels
    """
    lams = reference_set.lams
    label_names = reference_set.label_names
    coeffs_all, covs, scatters, chisqs, pivots, label_vector = model
    nlabels = len(pivots)

    # Baseline spectrum with continuum
    baseline_spec = coeffs_all[:,0]
    cont = np.round(baseline_spec,5) == 1
    baseline_spec = np.ma.array(baseline_spec, mask=cont)
    lams = np.ma.array(lams, mask=cont)
    fig, axarr = plt.subplots(2, sharex=True)
    plt.xlabel(r"Wavelength $\lambda (\AA)$")
    plt.xlim(np.ma.min(lams), np.ma.max(lams))
    ax = axarr[0]
    ax.plot(lams, baseline_spec, c='k', linewidth=0.3,
            label=r'$\theta_0$' + "= the leading fit coefficient")
    # contpix = list(np.loadtxt("contpix.txt"))
    contpix_lambda = list(np.loadtxt("contpix_lambda.txt", usecols=(0,),
                                     unpack=1))
    y = [1]*len(contpix_lambda)
    ax.scatter(contpix_lambda, y, s=1, color='r',label="continuum pixels")
    ax.legend(loc='lower right', prop={'family':'serif', 'size':'small'})
    ax.set_title("Baseline Spectrum with Continuum Pixels")
    ax.set_ylabel(r'$\theta_0$')
    ax = axarr[1]
    ax.plot(lams, baseline_spec, c='k', linewidth=0.3,
            label=r'$\theta_0$' + "= the leading fit coefficient")
    # contpix_lambda = list(np.loadtxt("contpix_lambda.txt",
    #     usecols = (0,), unpack =1))
    ax.scatter(contpix_lambda, y, s=1, color='r',label="continuum pixels")
    ax.set_title("Baseline Spectrum with Continuum Pixels, Zoomed")
    ax.legend(loc='upper right', prop={'family':'serif', 'size':'small'})
    ax.set_ylabel(r'$\theta_0$')
    ax.set_ylim(0.95, 1.05)
    filename = "baseline_spec_with_cont_pix.png"
    print("Diagnostic plot: fitted 0th order spectrum, cont pix overlaid.")
    print("Saved as %s" % filename)
    plt.savefig(filename)
    plt.close()

    # Leading coefficients for each label & scatter
    # Scale coefficients so that they can be overlaid on the same plot
    stds = np.array([np.std(coeffs_all[:,i+1]) for i in range(nlabels)])
    pivot_std = max(stds)
    ratios = np.round(pivot_std / stds, -1)  # round to the nearest 10
    ratios[ratios == 0] = 1
    fig, axarr = plt.subplots(2, sharex=True)
    plt.xlabel(r"Wavelength $\lambda (\AA)$")
    plt.xlim(np.ma.min(lams), np.ma.max(lams))
    ax = axarr[0]
    ax.set_ylabel("Leading coefficient " + r"$\theta_i$")
    ax.set_title("First-Order Fit Coefficients for Labels")
    lbl = r'$\theta_{0:d}=coeff for ${1:s}$ * {2:f}$'
    for i in range(nlabels):
        coeffs = coeffs_all[:,i+1] * ratios[i]
        # ax.plot(lams, coeffs, linewidth=0.3,
        #         label=r'$\theta_%s$'%(i+1) +'=coeff for ' + r" $%s$*%s"%(label_names[i], ratios[i]))
        ax.plot(lams, coeffs, linewidth=0.3,
                label=lbl.format(i+1, label_names[i], ratios[i]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
    ax.legend(bbox_to_anchor=(0., -.2, 1., .102), loc=3, ncol=3, mode="expand",
              prop={'family':'serif', 'size':'small'})
    ax = axarr[1]
    ax.set_ylabel("scatter")
    ax.set_title("Scatter of Fit")
    ax.plot(lams, scatters, c='k', linewidth=0.3)
    fig.tight_layout(pad=2.0, h_pad=4.0)
    print("Diagnostic plot: leading coeffs and scatters across wavelength.")
    filename = "leading_coeffs.png"
    print("Saved as %s" % filename)
    fig.savefig(filename)
    plt.close(fig)

    # Histogram of the chi squareds of ind. stars
    plt.hist(np.sum(chisqs, axis=0), color='lightblue', alpha=0.7)
    dof = len(lams) - coeffs_all.shape[1]   # for one star
    plt.axvline(x=dof, c='k', linewidth=2, label="DOF")
    plt.legend()
    plt.title("Distribution of " + r"$\chi^2$" + " of the Model Fit")
    plt.ylabel("Count")
    plt.xlabel(r"$\chi^2$" + " of Individual Star")
    filename = "modelfit_chisqs.png"
    print("Diagnostic plot: histogram of the red chi squareds of the fit")
    print("Saved as %s" % filename)
    plt.savefig(filename)
    plt.close()
