""" Plot the carbon and nitrogen theoretical gradient spectra,
overlaid with our "gradient spectra" from The Cannon """

import numpy as np
import matplotlib.pyplot as plt
import pyfits
import copy
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
from matplotlib import cm
from matplotlib.colors import LogNorm
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
    ivar = np.ones(spec.shape)*0.5
    cont = continuum_normalization._find_cont_gaussian_smooth(
            wl, spec, ivar, w)
    norm_flux, norm_ivar = continuum_normalization._cont_norm(
            spec, ivar, cont)
    return norm_flux[0]


def plot_cannon(ax, wl, grad_spec):
    ax.plot(
            wl, grad_spec, c='black', label="Cannon Gradient Spectrum", 
            linewidth=1, drawstyle='steps-mid')
    ax.legend(loc='lower left')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


def plot_model(ax, wl, grad_spec):
    ax.plot(
            wl, grad_spec, c='black', label="Model Gradient Spectrum", 
            linewidth=0.5, drawstyle='steps-mid')#, linestyle='-')
    ax.legend(loc='lower left')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


def gen_cannon_grad_spec(base_labels, choose, low, high, coeffs, pivots):
    """ Generate Cannon gradient spectra

    Parameters
    ----------
    labels: default values for [teff, logg, feh, cfe, nfe, afe, ak]
    choose: val of cfe or nfe, whatever you're varying
    low: lowest val of cfe or nfe, whatever you're varying
    high: highest val of cfe or nfe, whatever you're varying
    """
    # Generate Cannon gradient spectra
    low_lab = copy.copy(base_labels)
    low_lab[choose] = low
    lvec = (train_model._get_lvec(np.array([low_lab]), pivots))[0]
    model_low = np.dot(coeffs, lvec)
    high_lab = copy.copy(base_labels)
    high_lab[choose] = high
    lvec = (train_model._get_lvec(np.array([high_lab]), pivots))[0]
    model_high = np.dot(coeffs, lvec)
    grad_spec = (model_high - model_low) / (high - low)
    return grad_spec


def get_model_spec_martell():
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

    ind = np.where(np.logical_and(dat['Nfe']==0.6, dat['FeH']==-1.41))[0]
    cfe = dat['cfe'][ind]
    # only step from -0.4 to 0.4
    #dflux = cannon_normalize(dat[ind[-1]][3])-cannon_normalize(dat[ind[0]][3])
    dflux = cannon_normalize(dat[ind[9]][3])-cannon_normalize(dat[ind[5]][3])
    #dcfe = cfe[1]-cfe[0]
    dcfe = cfe[9]-cfe[5]
    c_grad_spec = (dflux/dcfe)

    ind = np.where(np.logical_and(dat['cfe']==-0.4, dat['FeH']==-1.41))[0]
    nfe = dat['nfe'][ind]
    # only step from -0.4 to 0.4
    dflux = cannon_normalize(dat[ind[5]][3])-cannon_normalize(dat[ind[1]][3])
    dnfe = nfe[5]-nfe[1]
    n_grad_spec = (dflux/dnfe)
    
    return wl, c_grad_spec, n_grad_spec


def get_model_spec_ting(atomic_number):
    """ 
    X_u_template[0:2] are teff, logg, vturb in km/s
    X_u_template[:,3] -> onward, put atomic number 
    atomic_number is 6 for C, 7 for N
    """
    DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"
    temp = np.load("%s/X_u_template_KGh_res=1800.npz" %DATA_DIR)
    X_u_template = temp["X_u_template"]
    wl = temp["wavelength"]
    if atomic_number == 6:
        print("Plotting Carbon")
    elif atomic_number == 7:
        print("Plotting Nitrogen")
    grad_spec = X_u_template[:,atomic_number]
    return wl, grad_spec
    #return wl, cannon_normalize(grad_spec)


if __name__=="__main__":
    DATA_DIR = "/Users/annaho/Data/LAMOST/Abundances"
    my_wl = np.load(DATA_DIR + "/" + "wl_cols.npz")['arr_0']
    label_names = np.load(DATA_DIR + "/" + "label_names.npz")['arr_0']

    m_coeffs = np.load(DATA_DIR + "/" + "model_0.npz")['arr_0']
    m_pivots = np.load(DATA_DIR + "/" + "model_0.npz")['arr_3']
    m_scatters = np.load(DATA_DIR + "/" + "model_0.npz")['arr_1']

    #labels = [4842, 2.97, -0.173, -0.3, -0.36, 0.046, 0.045]
    # Yuan-Sen: K giant, solar metallicity
    # labels = [4800, 2.5, 0, -0.1, 0.2, 0.05, 0.05]
    labels = [4800, 2.5, 0.03, 0.10, -0.17, -0.17, 0, -0.16,
            -0.13, -0.15, 0.13, 0.08, 0.17, -0.062]

def plot_cn():
    mg_grad_spec = gen_cannon_grad_spec(
            labels, 7, -0.5, 0.3, m_coeffs, m_pivots)
    c_grad_spec = gen_cannon_grad_spec(
            labels, 5, -0.3, 0.3, m_coeffs, m_pivots) 
    n_grad_spec = gen_cannon_grad_spec(
            labels, 4, -0.4, 0.4, m_coeffs, m_pivots) 

    wl, mg_grad_model = get_model_spec_ting(12)
    wl, c_grad_model = get_model_spec_ting(6)
    wl, n_grad_model = get_model_spec_ting(7)

    # Make a plot
    fig, (ax0,ax1) = plt.subplots(ncols=2, figsize=(12,6), 
                                  sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)
    x = 0.05
    y = 0.90
    ax0.text(
            x, y, r"-0.3 \textless [C/Fe] \textless 0.3", 
            transform=ax0.transAxes, fontsize=16)
    ax1.text(x, y, r"-0.4 \textless [N/Fe] \textless 0.4", 
            transform=ax1.transAxes, fontsize=16)

    plot_cannon(ax0, my_wl, c_grad_spec)
    plot_cannon(ax1, my_wl, n_grad_spec)

    plot_model(ax0, wl, c_grad_model-1)
    plot_model(ax1, wl, n_grad_model-1)

    ax0.set_xlim(4050,4400)
    ax1.set_xlim(4050, 4400)
    ax0.set_ylim(-0.4, 0.2)
    ax1.set_ylim(-0.4, 0.2)
    ax0.set_xlabel(r"Wavelength $\lambda (\AA)$", fontsize=18)
    ax1.set_xlabel(r"Wavelength $\lambda (\AA)$", fontsize=18)
    ax0.set_ylabel("Normalized Flux", fontsize=18)
    #ax0.axvline(x=4310, c='r')
    #ax1.axvline(x=4310, c='r')

    plt.show()
    #plt.savefig("cn_features.png")
