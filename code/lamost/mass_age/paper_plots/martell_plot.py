""" Reproduce Martell et al. 2008 figure """

import numpy as np
import matplotlib.pyplot as plt
import pyfits
from TheCannon import train_model
from TheCannon import continuum_normalization


DATA_DIR = "/Users/annaho/Data/Martell"
inputf = "ssg_wv.fits"
a = pyfits.open(DATA_DIR + "/" + inputf)
wl = a[1].data
a.close()

inputf = "ssg_nowv.fits"
a = pyfits.open(DATA_DIR + "/" + inputf)
dat = a[1].data
a.close()

# In the paper, the [C/Fe] goes from -1.4 to 0.4
# [Fe/H] = -1.41, [N/Fe] = 0.6
# There are 10 gridpoints in [C/Fe] and 14 in [N/Fe]

# In Martell 2008, spectra are normalized at 4319.2 Angstroms = ind=344
def normalize(spec):
    return spec / spec[344]

def cannon_normalize(spec_raw):
    spec = np.array([spec_raw])
    wl = np.arange(0, spec.shape[1])
    w = continuum_normalization.gaussian_weight_matrix(wl, L=50)
    ivar = np.ones(spec.shape)
    cont = continuum_normalization._find_cont_gaussian_smooth(
            wl, spec, ivar, w)
    norm_flux, norm_ivar = continuum_normalization._cont_norm(
            spec, ivar, cont)
    return norm_flux[0]

ind = np.where(np.logical_and(dat['Nfe']==0.6, dat['FeH']==-1.41))[0]
cfe = dat['cfe'][ind]
dflux = cannon_normalize(dat[ind[-1]][3])-cannon_normalize(dat[ind[0]][3])
#dflux = normalize(dat[ind[-1]][3]) - normalize(dat[ind[0]][3])
dcfe = cfe[-1]-cfe[0]
grad_spec = (dflux/dcfe)
plt.plot(
        wl, grad_spec, c='magenta', alpha=0.3, label="Martell Grad Spec for C",
        linewidth=2)

ind = np.where(np.logical_and(dat['cfe']==-0.4, dat['FeH']==-1.41))[0]
nfe = dat['nfe'][ind]
dflux = cannon_normalize(dat[ind[-1]][3])-cannon_normalize(dat[ind[0]][3])
#dflux = normalize(dat[ind[-1]][3]) - normalize(dat[ind[0]][3])
dnfe = nfe[-1]-nfe[0]
grad_spec = (dflux/dnfe)
plt.plot(
        wl, grad_spec, c='green', label="Martell Grad Spec for N",
        alpha=0.3, linewidth=2)

DATA_DIR = "/Users/annaho/Data/Mass_And_Age"
my_wl = np.load(DATA_DIR + "/" + "wl.npz")['arr_0']

coeffs = np.load(DATA_DIR + "/" + "coeffs.npz")['arr_0']
pivots = np.load(DATA_DIR + "/" + "pivots.npz")['arr_0']
scatter = np.load(DATA_DIR + "/" + "scatters.npz")['arr_0']
chisq = np.load(DATA_DIR + "/" + "chisqs.npz")['arr_0']
#cannon_label = np.load(
#        "%s/test_results_0.npz" %(DATA_DIR))['arr_0']
# In the paper, the [C/Fe] goes from -1.4 to 0.4
# [Fe/H] = -1.41, [N/Fe] = 0.6
# There are 10 gridpoints in [C/Fe] and 14 in [N/Fe]
# sample star: 4842, 2.97, -0.173, -0.3, 0.36, 0.046, 0.045
teff = 4000
logg = 1.3
feh = -1.4
nfe = 0.6
cfe = -0.4
afe = 0.26
ak = 0.0

low_c = np.array([teff, logg, feh, -0.3, nfe, afe, ak])
high_c = np.array([teff, logg, feh, 0.3, nfe, afe, ak])
lvec = (train_model._get_lvec(np.array([low_c]), pivots))[0]
model_low_c = np.dot(coeffs, lvec)
lvec = (train_model._get_lvec(np.array([high_c]), pivots))[0]
model_high_c = np.dot(coeffs, lvec)
c_grad_spec = (model_high_c - model_low_c) / (high_c[3] - low_c[3])

low_n = np.array([teff, logg, feh, cfe, -0.36, afe, ak])
high_n = np.array([teff, logg, feh, cfe, 0.36, afe, ak])
lvec = (train_model._get_lvec(np.array([low_n]), pivots))[0]
model_low_n = np.dot(coeffs, lvec)
lvec = (train_model._get_lvec(np.array([high_n]), pivots))[0]
model_high_n = np.dot(coeffs, lvec)
n_grad_spec = (model_high_n - model_low_n) / (high_n[4] - low_n[4])

#cn_coeffs = (coeffs[:,4])
#nm_coeffs = (coeffs[:,5])
plt.plot(
        my_wl, c_grad_spec/2, c='magenta', alpha=0.3, 
        #my_wl, cn_coeffs, c='magenta', alpha=0.7,
        label="Cannon Grad for C /2", linestyle='--', linewidth=2)
        #label="Cannon Leading Coeff for C", linestyle='--', linewidth=1)
plt.plot(
        my_wl, n_grad_spec/2, c='green', alpha=0.3, 
        #my_wl, nm_coeffs, c='green', alpha=0.7,
        label="Cannon Grad for N /2", linestyle='--', linewidth=2)
        #label="Cannon Leading Coeff for N", linestyle='--', linewidth=1)

#plt.plot(my_wl, (cn_coeffs+0.3)/2, c='r', lw=2)
plt.xlim(3900,4400)
plt.ylim(-0.2,0.2)

plt.legend(loc='lower left')

plt.show()
