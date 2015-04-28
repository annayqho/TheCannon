# loading Melissa's contpix

import numpy as np
mknpix = np.loadtxt("pixtest4.txt", dtype=int)
bad_pix = np.std(fluxes, axis=0) == 0 # gaps
mkn_contmask = np.zeros(len(bad_pix))
mkn_contmask[mknpix] = 1
mkn_contmask[bad_pix] = 0


# running contpix identifier

from apogee import ApogeeDataset

dataset = ApogeeDataset("example_DR10/Data",
                        "example_DR10/Data",
                        "example_DR10/reference_labels.csv")

cd cannon
from find_continuum_pixels import find_contpix_regions

contmask = find_contpix_regions(dataset.wl, dataset.tr_fluxes, dataset.tr_ivars, dataset.ranges)

# comparisons

npix = len(contmask)

#nocont = np.logical_and(mkn_contmask == 0, contmask==0) 
#hist(mkn_contmask[~nocont]-contmask[~nocont])

#same = mkn_contmask[~nocont] == contmask[~nocont]
#float(sum(same))/len(contmask[~nocont])
# 9.5% of the cont pixels are the same...oi

# continuum-normalize using the mask, and then make a new mask
from continuum_normalization import cont_norm, cont_norm_regions
norm_tr_fluxes, norm_tr_ivars = cont_norm_regions(
        dataset.tr_fluxes, dataset.tr_ivars, contmask, dataset.ranges)
ntrials=15
new_contmasks = np.zeros((ntrials, len(contmask)))
sample_spec = np.zeros((ntrials, len(contmask)))

for i in range(0, ntrials):
    new_contmask = find_contpix_regions(
            dataset.wl, norm_tr_fluxes, norm_tr_ivars, dataset.ranges)
    nocont = np.logical_and(contmask==0, new_contmask==0)
    #hist(contmask[~nocont]-new_contmask[~nocont])
    same = new_contmask[~nocont] == contmask[~nocont]
    print(sum(same))
    print(float(sum(same))/(npix-sum(nocont)))
    norm_tr_fluxes, norm_tr_ivars = cont_norm_regions(
            dataset.tr_fluxes, dataset.tr_ivars, new_contmask, dataset.ranges)
    new_contmasks[i] = new_contmask
    sample_spec[i] = norm_tr_fluxes[0,:]

sum(new_contmasks, axis=1)
x = np.linspace(0, ntrials-1, ntrials)
scatter(x[0:ntrials], sum(new_contmasks[0:ntrials], axis=1))
pix = np.linspace(0, npix-1, npix)

for i in range(0, ntrials):
    scatter(pix, new_contmasks[i]*i, s=2)

for i in range(0, ntrials):
    plot(sample_spec[i]+i)
