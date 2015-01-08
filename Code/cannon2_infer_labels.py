"""This runs Step 2 of The Cannon:
    uses the model to solve for the labels of the test set."""

from scipy import optimize as opt
import numpy as np

def get_x(labels):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels
    """
    nlabels = len(labels)
    x = labels # linear term
    # Quadratic terms: 
    for i in range(nlabels):
        for j in range(i, nlabels):
            element = labels[i]*labels[j]
            x.append(element)
    x = np.array(x)
    return x

# Extremely annoying...looks like scipy_optimize can't handle an array of parameters. Have to spell all the parameters out. 
# So for now we have to hard-code the number of labels into here.  
def func(coeffs, a, b, c, d):
    labels = [a,b,c,d]
    x = get_x(labels)
    return np.dot(coeffs, x)

def infer_labels(num_labels, model, test_set):
    """
    Uses the model to solve for labels of the test set.

    Input:
    -----
    num_labels: number of labels being solved for
    model: coeffs_all, covs, scatters, chis, chisqs, pivots
    test_set: Stars object (see stars.py) corresponding to the test set. 

    Returns
    -------
    labels_all: 
    MCM_rotate_all:
    covs_all: covariance matrix of the fit
    """
    nlabels = num_labels
    spectra = test_set.get_spectra() #(nstars, npixels, 3)
    nstars = spectra.shape[0]
    npixels = spectra.shape[1]
    coeffs_all, covs, scatters, chis, chisqs, pivots = model
    labels_all = np.zeros((nstars, nlabels))
    # Don't understand what this MCM_rotate_all matrix is
    MCM_rotate_all = np.zeros((nstars, coeffs_all.shape[1]-1, 
        coeffs_all.shape[1]-1.))
    covs_all = np.zeros((nstars, nlabels, nlabels))

    for jj in range(nstars):
        pixels = spectra[jj,:,0]
        fluxes = spectra[jj,:,1]
        fluxerrs = spectra[jj,:,2]
        fluxes_norm = fluxes - coeffs_all[:,0] # ? code says "subtract the mean". but in the code it's written here coeffs[:,0] and I don't see why that should be the mean. I also don't see why we should subtract the mean...
        # We have 15 coefficients in coeffs...
        Cinv = 1. / (fluxerrs* 2 + scatters**2)
        # something I don't understand --
        coeffs = np.delete(coeffs_all, 0, axis=1) #since x0 = 1...
        weights = 1 / Cinv**0.5
        labels, covs = opt.curve_fit(
                func, coeffs, fluxes_norm, sigma=weights, absolute_sigma = True)
        labels = labels + pivots
        value_cut = -14 # no idea what this is...
        coeffs_slice = coeffs[:,value_cut:]
        MCM_rotate = np.dot(coeffs_slice.T, Cinv[:,None] * coeffs_slice)
        labels_all[jj,:] = labels
        MCM_rotate_all[jj,:,:] = MCM_rotate
        covs_all[jj,:,:] = covs
    return labels_all, MCM_rotate_all, covs_all
