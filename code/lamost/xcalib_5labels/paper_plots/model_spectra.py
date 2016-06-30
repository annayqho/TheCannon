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

def spectral_model():
    # We've left out group 0
    group = 0

    groups = np.load("../ref_groups.npz")['arr_0']
    choose = groups == group

    direc = "/Users/annaho/TheCannon/data/lamost_paper"
    wl = np.load("%s/wl.npz" %direc)['arr_0']
    ref_id = np.load("%s/ref_id.npz" %direc)['arr_0']
    flux = np.load("%s/ref_flux.npz" %direc)['arr_0'][choose]
    ivar = np.load("%s/ref_ivar.npz" %direc)['arr_0'][choose]
    snr = np.load("%s/ref_snr.npz" %direc)['arr_0'][choose]

    scat = np.load("../ex%s_scatters.npz" %group)['arr_0']
    coeffs = np.load("../ex%s_coeffs.npz" %group)['arr_0']
    pivots = np.load("../ex%s_pivots.npz" %group)['arr_0']
    cannon_label = np.load("../ex%s_cannon_label_vals.npz" %group)['arr_0']
    lvec_all = train_model._get_lvec(cannon_label, pivots)

    #xmin = min(wl)
    #xmax = max(wl)
    xmin = 5000
    xmax = 5340

    r_ymin = -0.05
    r_ymax = 0.05
    ymin = 0.6
    ymax = 1.15

    ii = 40
    f = flux[ii,:]
    iv = ivar[ii,:]
    label = cannon_label[ii,:]
    lvec = lvec_all[ii,:]
    model = np.dot(coeffs, lvec)

    # err = scat ^2 + uncertainty^2

    iv_tot = (iv/(scat**2 * iv + 1))
    err = np.ones(len(iv_tot))*1000
    err[iv_tot>0] = 1/iv_tot[iv_tot>0]**0.5

    # Cinv = ivars / (1 + ivars*scatter**2)
    # lTCinvl = np.dot(lvec.T, Cinv[:, None] * lvec)
    # lTCinvf = np.dot(lvec.T, Cinv * fluxes)

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
    plt.show()
    #plt.savefig("model_spectrum_full.png")
    #plt.savefig("model_spectrum.png")


if __name__=="__main__":
    spectral_model()
