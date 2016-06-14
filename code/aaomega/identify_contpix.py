""" 
Use the model fit from pseudo continuum normalized spectra 
to identify continuum pixels. 

From Ness et al. 2015:

About 35% of the pixels in the resulting baseline spectrum 
(the vector q 0) have flux levels within 1% of unity. 
However, not all these pixels are suitable continuum pixels, 
as many of them have significant dependencies, q1,2,3, 
on the three labels. 
In practice, a good set of continuum pixels can be identified from the
APOGEE spectra using a flux cut in the baseline spectra of the
model, 1 ± 0.15 (0.985–1.015), combined with the smallest
20–30 percentile of the first order coefficients, q1,2,3, which l
retains between 5% and 9% of pixels. 
We found empirically that changing the latter percentiles to 
(q1,q2,q3) <(1e−5, 0.0045, 0.0085) returns only 6.5% of the pixels, 
but ultimately makes for an even better match to the ASPCAPlabel scale; 
we adopt this procedure. 
We use the inverse variance weighting of these pixels for the 
corresponding second order Chebyshev polynomial fit, 
adding an additional error term that is set to 0 for continuum pixels 
and a large error value for all other pixels so that the new error term 
s¢ for each pixel becomes: s¢= s + s .
"""

import numpy as np
import matplotlib.pyplot as plt

def get_perc(array, percentile):
    num = len(array)
    order = np.argsort(np.abs(array))
    sorted = array[order]
    smallest_quarter = order[0:int(percentile*num)]
    choose = np.zeros(num, dtype=bool)
    choose[smallest_quarter] = 1
    return choose

wl = np.load("wl.npz")['arr_0']
coeffs = np.load("pseudo_model_coeffs.npz")['arr_0']
npix = coeffs.shape[0]

# What fraction of pixels in the baseline spectrum are within 1% of unity?
baseline_spec = coeffs[:,0]
unity = 1.0 - baseline_spec < 0.013

# 24% of pixels are within 1% of unity.

# What fraction of these pixels are in the smallest 30th percentile of the
# leading Teff coefficient?
teff = coeffs[:,1]
teff_choose = get_perc(teff, 0.3)

# logg
logg = coeffs[:,2]
logg_choose = get_perc(logg, 0.4)

feh = coeffs[:,3]
feh_choose = get_perc(feh, 0.4)

choose1 = np.logical_and(unity, teff_choose)
choose2 = np.logical_and(logg_choose, feh_choose)
choose = np.logical_and(choose1, choose2)
print(sum(choose)/len(choose))
# 9.6%

plt.plot(wl, baseline_spec, c='k')
plt.plot(wl, teff*2000+0.5, c='k')
plt.plot(wl, logg*10, c='k')
plt.plot(wl, feh-0.2, c='k')
wl_choose = wl[choose]
np.savez("wl_contpix.npz", wl_choose)
for val in wl_choose: plt.axvline(x=val, c='r')
plt.scatter(wl[unity],baseline_spec[unity], c='r')
plt.scatter(wl[teff_choose],(teff*2000+0.5)[teff_choose], c='r')
plt.scatter(wl[logg_choose], (logg*10)[logg_choose], c='r')
plt.scatter(wl[feh_choose], (feh-0.2)[feh_choose], c='r')
plt.show()

