""" Plot the carbon and nitrogen theoretical gradient spectra,
overlaid with our "gradient spectra" from The Cannon """

import numpy as np
import matplotlib.pyplot as plt
import pyfits
import copy
from matplotlib import rc
from scipy import interpolate 
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


def resample(grid, wl, flux):
    """ Resample spectrum onto desired grid """
    flux_rs = (interpolate.interp1d(wl, flux))(grid)
    return flux_rs


def plot_cannon(ax, wl, grad_spec):
    ax.plot(
            wl, grad_spec, c='black', label="Cannon Gradient Spectrum", 
            linewidth=0.5, drawstyle='steps-mid')
    ax.legend(loc='lower left')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


def plot_model(ax, wl, grad_spec):
    ax.plot(
            wl, grad_spec, c='red', label="Model Gradient Spectrum", 
            linewidth=0.5, drawstyle='steps-mid')#, linestyle='-')
    ax.legend(loc='lower left')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


def gen_cannon_grad_spec(choose, coeffs, pivots):
    """ Generate Cannon gradient spectra

    Parameters
    ----------
    labels: default values for [teff, logg, feh, cfe, nfe, afe, ak]
    choose: val of cfe or nfe, whatever you're varying
    low: lowest val of cfe or nfe, whatever you're varying
    high: highest val of cfe or nfe, whatever you're varying
    """
    base_labels = [4800, 2.5, 0.03, 0.10, -0.17, -0.17, 0, -0.16,
            -0.13, -0.15, 0.13, 0.08, 0.17, -0.062]
    label_names = np.array(
            ['TEFF', 'LOGG', 'AK', 'Al', 'Ca', 'C', 'Fe', 'Mg', 'Mn',
            'Ni', 'N', 'O', 'Si', 'Ti'])
    label_atnum = np.array(
            [0, 1, -1, 13, 20, 6, 26, 12, 25, 28, 7, 8, 14, 22])
    # Generate Cannon gradient spectra
    ind = np.where(label_atnum==choose)[0][0]
    low_lab = copy.copy(base_labels)
    high = base_labels[ind]
    if choose > 0:
        low = base_labels[ind] - 0.2
    else: #temperature
        if choose != 0: print("warning...")
        low = base_labels[ind] - 200
    low_lab[ind] = low
    lvec = (train_model._get_lvec(np.array([low_lab]), pivots))[0]
    model_low = np.dot(coeffs, lvec)
    lvec = (train_model._get_lvec(np.array([base_labels]), pivots))[0]
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
    grad_spec = X_u_template[:,atomic_number]
    return wl, grad_spec


def get_cannon_model():
    DATA_DIR = "/Users/annaho/Data/LAMOST/Abundances"
    my_wl = np.load(DATA_DIR + "/" + "wl_cols.npz")['arr_0']
    label_names = np.load(DATA_DIR + "/" + "label_names.npz")['arr_0']

    m_coeffs = np.load(DATA_DIR + "/" + "model_0.npz")['arr_0']
    m_pivots = np.load(DATA_DIR + "/" + "model_0.npz")['arr_3']
    m_scatters = np.load(DATA_DIR + "/" + "model_0.npz")['arr_1']
    return my_wl, label_names, m_coeffs, m_pivots, m_scatters


def plot_info_content(ax, grad_spec, label):
    rms = grad_spec**2
    rms_norm = rms/sum(rms)
    order = np.argsort(rms_norm)[::-1]
    rms_norm_sorted = rms_norm[order]
    foo = np.cumsum(rms_norm_sorted)
    nelem = len(foo)
    mark = np.where(foo > 0.8)[0][0]
    imp = order[0:mark]
    print(len(imp)/len(order))
    xvals = np.linspace(0,1,nelem)
    ax.plot(xvals, foo, c='black', linestyle='--')
    ax.axvline(x=xvals[mark], c='r')
    ax.set_title("Information Content for: " + label)
    ax.set_xlabel("Fraction of Spectral Pixels")
    ax.set_ylabel("Fraction of Information Content")
    return imp # index of pix that contain 80% of the info


if __name__=="__main__":
    label_names = np.array(
            ['TEFF', 'LOGG', 'AK', 'Al', 'Ca', 'C', 'Fe', 'Mg', 'Mn',
            'Ni', 'N', 'O', 'Si', 'Ti'])
    label_atnum = np.array(
            [0, 1, -1, 13, 20, 6, 26, 12, 25, 28, 7, 8, 14, 22])
    my_wl, label_names, m_coeffs, m_pivots, m_scatters = get_cannon_model()
    npix = len(my_wl)
    for ii,label_name in enumerate(label_names):
        filt = np.zeros(npix, dtype=bool)
        #label_name = "Ti"
        #ind = 22 # atomic number (0 for temp, 1 for logg)
        label_name = label_name.decode("utf-8")
        print(label_name)
        ind = label_atnum[ii]
        if ind==-1:
            # for Ak, save a filter that's just the colours
            filt[3626:] = 1
            np.savez("%s_filter.npz" %label_name, (my_wl, filt))
        else:
            wl, grad_spec_raw = get_model_spec_ting(ind)
            grad_spec_rs = resample(my_wl, wl, grad_spec_raw)
            grad_spec = cannon_normalize(grad_spec_rs+1)-1
            cannon_grad_spec = gen_cannon_grad_spec(ind, m_coeffs, m_pivots)
            fig, ax = plt.subplots(1, figsize=(15,6))
            imp_pix = plot_info_content(ax, grad_spec, label_name)
            plt.savefig("%s_info_content.png" %label_name)
            plt.close()
            filt[imp_pix] = 1
            fig, ax = plt.subplots(1, figsize=(15,6))
            plot_model(ax, my_wl, grad_spec)
            ax.scatter(my_wl[filt], grad_spec[filt], s=5, lw=0, c='k')
            #plot_cannon(ax, my_wl, cannon_grad_spec)
            plt.ylim(min(grad_spec), max(grad_spec))
            plt.xlim(min(wl),max(wl))
            plt.xlabel("Wavelength (Angstroms)", fontsize=20)
            plt.ylabel("dFlux/d%s" %label_name, fontsize=20)
            np.savez("%s_filter.npz" %label_name, (my_wl, filt))
            plt.savefig("%s_filter.png" %label_name)
            plt.close()
        #     # plt.show()
