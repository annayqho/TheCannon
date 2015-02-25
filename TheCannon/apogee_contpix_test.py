import pickle
dataset = pickle.load(open("dataset.pickle", "rb"))
import numpy as np
mknpix = np.loadtxt("pixtest4.txt", dtype=int)
fluxes = dataset.tr_fluxes
bad_pix = np.std(fluxes, axis=0) == 0 # gaps
mkn_contmask = np.zeros(len(bad_pix))
mkn_contmask[mknpix] = 1
mkn_contmask[bad_pix] = 0

npix = len(dataset.contmask)
pix = np.linspace(0, npix-1, npix)

# compare my contmask to Melissa's
hist(mkn_contmask-dataset.contmask)
diff=np.abs(mkn_contmask-dataset.contmask)
# mine has 377, Melissa's has 460
same = mkn_contmask == dataset.contmask
float(sum(same))/len(dataset.contmask)
# 701 pixels are different, which is a lot considering there are only 
# 837 max possible continuum pixel locations...
# 0.9182507288629738

scatter(pix, dataset.contmask+0.1, c='r')
scatter(pix, mkn_contmask, c='b')

