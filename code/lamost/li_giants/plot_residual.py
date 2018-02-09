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
#from TheCannon import train_model
from TheCannon import dataset
from TheCannon import model
from matplotlib.ticker import MaxNLocator


def plot(ii, wl, flux, ivar, model_all, coeffs, scatters, chisqs, pivots, start_wl, end_wl, highlights, figname):
    xmin = start_wl
    xmax = end_wl 

    f = flux[ii,:]
    iv = ivar[ii,:]
    model_spec = model_all[ii,:]
    choose = np.logical_and(wl > xmin, wl < xmax)
    ymin = min(f[choose])-0.05
    ymax = max(f[choose])+0.05
    
    # err = scat ^2 + uncertainty^2
    m = model.CannonModel(2, useErrors = False)
    m.coeffs = coeffs
    m.scatters = scatters
    m.chisqs = chisqs
    m.pivots = pivots

    scat = m.scatters
    iv_tot = (iv/(scat**2 * iv + 1))
    err = np.ones(len(iv_tot))*1000
    err[iv_tot>0] = 1/iv_tot[iv_tot>0]**0.5

    #print("X2 is: " + str(sum((f - model_spec)**2 * iv_tot)))

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
    #ax_spectrum.scatter(wl, f, c='k')
    ax_spectrum.plot(
            wl, model_spec, c='r', alpha=0.7, label="The Cannon Model")
    ax_spectrum.fill_between(
            wl, model_spec+err, model_spec-err, alpha=0.1, color='r')
    ax_spectrum.set_ylim(ymin, ymax)
    ax_spectrum.set_xlim(xmin, xmax)
    ax_spectrum.axhline(1, c="k", linestyle=":", zorder=-1)
    ax_spectrum.legend(loc="lower right")

    resid = f-model_spec
    r_ymin = min(resid[choose])-0.01
    r_ymax = max(resid[choose])+0.01
    ax_residual.plot(wl, resid, c='k', alpha=0.8, drawstyle='steps-mid')
    ax_residual.fill_between(wl, resid+err, resid-err, alpha=0.1, color='k')
    ax_residual.set_ylim(r_ymin,r_ymax)
    ax_residual.set_xlim(ax_spectrum.get_xlim())
    ax_residual.axhline(0, c="k", linestyle=":", zorder=-1)
    for highlight in highlights:
        ax_residual.axvline(x=highlight, c='r', linewidth=2, linestyle='--')
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
    for highlight in highlights:
        plt.axvline(x=highlight, c='r', linewidth=2, linestyle='--')
    #plt.show()
    plt.savefig(figname)
    plt.close()
    #plt.savefig("model_spectrum_full.png")
    #plt.savefig("model_spectrum.png")
