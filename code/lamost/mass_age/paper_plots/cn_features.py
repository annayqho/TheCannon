""" Plot the carbon and nitrogen theoretical gradient spectra,
overlaid with our "gradient spectra" from The Cannon """

import numpy as np
import matplotlib.pyplot as plt
import pyfits
from TheCannon import train_model
from TheCannon import continuum_normalization


def normalize(spec):
    """ Normalize according to Martell et al 2008
    That is to say, spectra are normalized at 4319.2 Angstroms
    """
    return spec / spec[344]


def cannon_normalize(spec_raw):
    """ Normalize according to The Cannon """
    spec = np.array([spec_raw])
    wl = np.arange(0, spec.shape[1])
    w = continuum_normalization.gaussian_weight_matrix(wl, L=50)
    ivar = np.ones(spec.shape)
    cont = continuum_normalization._find_cont_gaussian_smooth(
            wl, spec, ivar, w)
    norm_flux, norm_ivar = continuum_normalization._cont_norm(
            spec, ivar, cont)
    return norm_flux[0]


def plot_panel(wl, grad_spec, label):
    plt.plot(
            wl, grad_spec, c='magenta', alpha=0.3, 
            label=label, linewidth=2)
    plt.xlim(3900,4400)
    plt.ylim(-0.2,0.2)
    plt.legend(loc='lower left')


def gen_cannon_grad_spec(labels, choose, low, high, coeffs, pivots):
    """ Generate Cannon gradient spectra

    Parameters
    ----------
    labels: default values for [teff, logg, feh, cfe, nfe, afe, ak]
    choose: val of cfe or nfe, whatever you're varying
    low: lowest val of cfe or nfe, whatever you're varying
    high: highest val of cfe or nfe, whatever you're varying
    """
    # Generate Cannon gradient spectra

    low_lab = labels
    low_lab[choose] = low
    lvec = (train_model._get_lvec(np.array([low_lab]), pivots))[0]
    model_low = np.dot(coeffs, lvec)
    high_lab = labels
    high_lab[choose] = high
    lvec = (train_model._get_lvec(np.array([high_lab]), pivots))[0]
    model_high = np.dot(coeffs, lvec)
    grad_spec = (model_high - model_low) / (high - low)

    return grad_spec

DATA_DIR = "/Users/annaho/Data/Mass_And_Age"
my_wl = np.load(DATA_DIR + "/" + "wl.npz")['arr_0']

m_coeffs = np.load(DATA_DIR + "/" + "coeffs.npz")['arr_0']
m_pivots = np.load(DATA_DIR + "/" + "pivots.npz")['arr_0']
#scatter = np.load(DATA_DIR + "/" + "scatters.npz")['arr_0']
#chisq = np.load(DATA_DIR + "/" + "chisqs.npz")['arr_0']

labels = [4842, 2.97, -0.173, -0.3, -0.36, 0.046, 0.045]
c_grad_spec = gen_cannon_grad_spec(
        labels, 3, -0.3, 0.3, m_coeffs, m_pivots)
n_grad_spec = gen_cannon_grad_spec(
        labels, 4, -0.36, 0.36, m_coeffs, m_pivots)
plot_panel(my_wl, c_grad_spec/2, label="Cannon Grad for C/2")
plot_panel(my_wl, n_grad_spec/2, label="Cannon grad for N/2")

# Carbon and nitrogen theoretical gradient spectra
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
ind = np.where(np.logical_and(dat['Nfe']==0.6, dat['FeH']==-1.41))[0]
cfe = dat['cfe'][ind]
dflux = cannon_normalize(dat[ind[-1]][3])-cannon_normalize(dat[ind[0]][3])
dcfe = cfe[-1]-cfe[0]
grad_spec = (dflux/dcfe)
plot_panel(wl, grad_spec, label="Martell Grad Spec for C")

ind = np.where(np.logical_and(dat['cfe']==-0.4, dat['FeH']==-1.41))[0]
nfe = dat['nfe'][ind]
dflux = cannon_normalize(dat[ind[-1]][3])-cannon_normalize(dat[ind[0]][3])
dnfe = nfe[-1]-nfe[0]
grad_spec = (dflux/dnfe)
plot_panel(wl, grad_spec, label="Martell Grad Spec for C")


plt.show()
