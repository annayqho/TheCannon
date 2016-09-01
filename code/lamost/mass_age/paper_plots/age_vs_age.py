import pyfits
from matplotlib import rc
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
import sys
import matplotlib.gridspec as gridspec
#sys.path.append("/Users/annaho/Dropbox/Research/TheCannon/code/lamost/mass_age/cn")
#from estimate_age import estimate_age

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/xval_with_cuts"
ids = np.load(DATA_DIR + "/ref_id.npz")['arr_0']
ages, age_errs = estimate_age()
a = pyfits.open(
    "/Users/annaho/Data/LAMOST/Mass_And_Age/age_vs_age_catalog.fits")
data = a[1].data
a.close()
id_all = data['lamost_id']
id_all = np.array(id_all)
id_all = np.array([val.strip() for val in id_all])
keep = np.in1d(ids, id_all)
ids = ids[keep]
ln_ages = np.log(10**ages)[keep]
good = age_errs[keep] < 1
inds = np.array([np.where(id_all==val)[0][0] for val in ids])
apogee_id = data['2mass'][inds]
ness_age = data['lnAge'][inds]

plt.hist2d(ness_age[good], ln_ages[good], bins=50, norm=LogNorm(), cmap="gray_r")
plt.plot([-2,5], [-2,5], c='k')
plt.xlabel("ln(Age) from APOGEE Spectroscopic Mass + Isochrones", fontsize=16)
plt.ylabel("ln(Age) from LAMOST [C/Fe] and [N/Fe]", fontsize=16)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(-2,4)
plt.ylim(-2,4)
plt.savefig("age_vs_age.png")
