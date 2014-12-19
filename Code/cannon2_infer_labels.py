# This is Step 2 of the Cannon.

from scipy import optimize as opt
import numpy as np

nlabels = 0
nstars = 0
npixels = 0

def get_x(labels):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels
    """
    x = labels # linear term
    # Quadratic terms: 
    for i in range(nlabels):
        for j in range(i, nlabels):
            element = labels[i]*labels[j]
            x.append(element)
    x = np.array(x)
    return x

# Extremely annoying...don't know if scipy_optimize can handle an array of parameters instead of spelling all the parameters out. So for now we have to hard-code the number of labels itno here.  
def func(coeffs, a, b, c, d):
    # labels = [a, b, c, d]
    labels = [a,b,c,d]
    x = get_x(labels)
    return np.dot(coeffs, x)

    # What was the point of ever having an x0 in the coeffs array?

def infer_labels(num_labels, model, test_set):
    """
    This determines the new labels from the model.
    """
    global nlabels
    nlabels = num_labels
    spectra = test_set.get_spectra() #(nstars, npixels, 3)
    global nstars
    nstars = spectra.shape[0]
    global npixels
    npixels = spectra.shape[1]

    coeffs_all, covs, scatters, chis, chisqs, pivots = model

    labels_all = np.zeros((nstars, nlabels))
    # Not sure what the MCM_rotate_all matrix is...
    MCM_rotate_all = np.zeros((nstars, coeffs_all.shape[1]-1, coeffs_all.shape[1]-1.))
    covs_all = np.zeros((nstars, nlabels, nlabels))

    for jj in range(nstars):
        pixels = spectra[jj,:,0]
        fluxes = spectra[jj,:,1]
        fluxerrs = spectra[jj,:,2]
        fluxes_norm = fluxes - coeffs_all[:,0] # ? code says "subtract the mean". but in the code it's written here coeffs[:,0] and I don't see why that should be the mean. I also don't see why we should subtract the mean...
        # We have 15 coefficients in coeffs...one for each star
        Cinv = 1. / (fluxerrs* 2 + scatters**2)

        coeffs = np.delete(coeffs_all, 0, axis=1) #x0 = 1...I don't understand this
        weights = 1 / Cinv**0.5
        # coeffs.shape = 8575, 14
        # opt.curve_fit(f, xdata, ydata, p0=None, sigma=None)
        # f = the model function, must take the independent variable as the first argument and the parameters to fit as separate remaining arguments.
        # assuems ydata = f(xdata, *params) + eps
        
        labels, covs = opt.curve_fit(func, coeffs, fluxes_norm, sigma=weights, absolute_sigma = True)
        labels = labels + pivots
        value_cut = -14 # no idea what this is...
        coeffs_slice = coeffs[:,value_cut:]
        MCM_rotate = np.dot(coeffs_slice.T, Cinv[:,None] * coeffs_slice)
        labels_all[jj,:] = labels
        MCM_rotate_all[jj,:,:] = MCM_rotate
        covs_all[jj,:,:] = covs

    return labels_all, MCM_rotate_all, covs_all
