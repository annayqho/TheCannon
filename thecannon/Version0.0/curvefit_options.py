from scipy import optimize as opt
import numpy as np

def get_x(labels):
    nlabels = len(labels)
    x = labels
    for i in range(nlabels):
        for j in range(i, nlabels):
            element = labels[i]*labels[j]
            x.append(element)
    x = np.array(x)
    return x

def func(coeffs, a, b, c, d):
    labels = [a, b, c, d]
    x = get_x(labels)
    return np.dot(coeffs, x)

# for one star:

labels, covs = opt.curve_fit(
    func, coeffs, fluxes_norm, sigma=weights, absolute_sigma = True)

# Options:

# We could either feed in a p0 matrix, generated using the training labels (like, each value is the average of the training labels). Or we could dynamically define func...which sounds really terrible. My inclination for "good coding" is to input guess values, but I don't know whether we can do this scientifically. I'll have to ask.  
