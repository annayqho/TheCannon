""" Create some model spectra vs. data """

import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
# rc('text.latex', preamble = ','.join('''\usepackage{txfonts}'''.split()))
plt.rc('font', family='serif')
from TheCannon import train_model
from matplotlib.ticker import MaxNLocator

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"

dibs = [4066, 4180, 4428, 4502, 4726, 4760, 4763, 4780, 4882, 4964, 
        5173, 5404, 5488, 5494, 5508, 5528, 5545, 5705, 5711, 5778, 
        5780, 5797, 5844, 5850, 
        6010, 6177, 6196, 6203, 6234, 6270, 6284, 6376, 6379, 6445, 6533, 
        6614, 6661, 6699, 6887, 6919, 6993, 7224, 7367, 7562, 8621]


def spectral_model(lamost_id, xmin, xmax):
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    ref_id = np.load("%s/ref_id.npz" %DATA_DIR)['arr_0']
    flux = np.load("%s/ref_flux.npz" %DATA_DIR)['arr_0']
    ivar = np.load("%s/ref_ivar.npz" %DATA_DIR)['arr_0']
    snr = np.load("%s/ref_snr.npz" %DATA_DIR)['arr_0']

    coeffs = np.load("%s/coeffs.npz" %DATA_DIR)['arr_0']
    scat = np.load("%s/scatters.npz" %DATA_DIR)['arr_0']
    chisqs = np.load("%s/chisqs.npz" %DATA_DIR)['arr_0']
    pivots = np.load("%s/pivots.npz" %DATA_DIR)['arr_0']
    cannon_label = np.load("%s/xval_cannon_label_vals.npz" %DATA_DIR)['arr_0']
    lvec_all = train_model._get_lvec(cannon_label, pivots)

    r_ymin = -0.06
    r_ymax = 0.06
    ymin = 0.75
    ymax = 1.1

    ii = np.where(ref_id==lamost_id)[0][0]
    
    f = flux[ii,:]
    iv = ivar[ii,:]
    label = cannon_label[ii,:]
    lvec = lvec_all[ii,:]
    model = np.dot(coeffs, lvec)

    # err = scat ^2 + uncertainty^2

    iv_tot = (iv/(scat**2 * iv + 1))
    err = np.ones(len(iv_tot))*1000
    err[iv_tot>0] = 1/iv_tot[iv_tot>0]**0.5

    print("X2 is: " + str(sum((f - model)**2 * iv_tot)))

    # Thanks to David Hogg / Andy Casey for this...
    # I stole it from the Annie's Lasso Github.
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,4])
    fig = plt.figure(figsize=(13.3, 4))
    ax_residual = plt.subplot(gs[0])
    ax_spectrum = plt.subplot(gs[1])

    ax_spectrum.plot(
            wl, f, c='k', alpha=0.7, drawstyle='steps-mid', label="Data")
    ax_spectrum.plot(
            wl, model, c='r', alpha=0.7, label="The Cannon Model")
    ax_spectrum.fill_between(
            wl, model+err, model-err, alpha=0.1, color='r')
    for dib in dibs:
        ax_spectrum.axvline(x=dib, c='k', lw=0.5)
    ax_spectrum.set_ylim(ymin, ymax)
    ax_spectrum.set_xlim(xmin, xmax)
    ax_spectrum.axhline(1, c="k", linestyle=":", zorder=-1)
    ax_spectrum.legend(loc="lower right")

    resid = f-model
    ax_residual.plot(wl, resid, c='k', alpha=0.8, drawstyle='steps-mid')
    ax_residual.fill_between(wl, resid+err, resid-err, alpha=0.1, color='k')
    ax_residual.set_ylim(r_ymin,r_ymax)
    ax_residual.set_xlim(ax_spectrum.get_xlim())
    ax_residual.axhline(0, c="k", linestyle=":", zorder=-1)
    ax_residual.set_xticklabels([])

    ax_residual.yaxis.set_major_locator(MaxNLocator(3))
    ax_residual.xaxis.set_major_locator(MaxNLocator(6))
    ax_spectrum.xaxis.set_major_locator(MaxNLocator(6))
    ax_spectrum.yaxis.set_major_locator(MaxNLocator(4))
    ax_spectrum.set_xlabel(r"Wavelength $\lambda (\AA)$", fontsize=18)
    ax_spectrum.set_ylabel("Normalized flux", fontsize=18)
    ax_spectrum.tick_params(axis="both", labelsize=18)
    ax_residual.tick_params(axis="both", labelsize=18)

    fig.tight_layout()
    return resid, fig
    #plt.show()
    #plt.savefig("model_spectrum_full.png")
    #plt.savefig("model_spectrum_%s_%s.png" %(ii, xmin))
    #plt.close()


if __name__=="__main__":
    spectral_model(117, 4000, 5000)
    #for ii in range(0, 50):
    #    spectral_model(ii, 5100, 5300)
