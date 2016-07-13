import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
from matplotlib.colors import LogNorm
from math import log10, floor
from TheCannon import train_model
import matplotlib.gridspec as gridspec


DATA_DIR = '/Users/annaho/Data/AAOmega'


def round_sig(x, sig=2):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)

def validation():
    orig = np.load("%s/val_label.npz" %DATA_DIR)['arr_0']
    cannon = np.load("%s/val_labels.npz" %DATA_DIR)['arr_0']
    snr = np.load("%s/val_SNR.npz" %DATA_DIR)['arr_0']
    choose = snr > 50

    labels = ["$\mathrm{T}_{\mathrm{eff}}$ (K)", "$\log g$ (dex)", "[M/H]"]
    mins = [4000, 0.5, -2.5]
    maxs = [5500, 3.5, 0.0]

    fig,axarr = plt.subplots(3,1, figsize=(4,10))#, sharey=True)
    props = dict(boxstyle='round', facecolor='white')

    for i in range(0,3):
        diff = cannon[:,i][choose] - orig[:,i][choose]
        bias = np.mean(diff)
        scat = np.std(diff)
        text1 = "Bias: %s\nRMS Scatter: %s" %(str(round_sig(bias)), str(round_sig(scat)))
        axarr[i].hist2d(
                orig[:,i][choose], cannon[:,i][choose], bins=20,
                range=[[mins[i], maxs[i]], [mins[i], maxs[i]]],
                cmap = "gray_r", norm=LogNorm())
        axarr[i].text(
                0.05, 0.95, text1, horizontalalignment='left',
                verticalalignment='top', transform=axarr[i].transAxes, bbox=props)
        axarr[i].plot(
                [mins[i],maxs[i]],[mins[i],maxs[i]], c='k', 
                linestyle='--', label="x=y")
        axarr[i].set_xlabel(labels[i]+" from Orig. Pipeline", fontsize=16)
        axarr[i].set_ylabel(labels[i]+" from Cannon", fontsize=16)
        axarr[i].set_xlim(mins[i], maxs[i])
        axarr[i].set_ylim(mins[i], maxs[i])
        #axarr[i].legend()
        #axarr[i].set_colorbar()
    fig.tight_layout()
    #plt.show()
    plt.savefig("1to1_validation.png")

def snr_dist():
    fig = plt.figure(figsize=(6,4))
    tr_snr = np.load("../tr_SNR.npz")['arr_0']
    snr = np.load("../val_SNR.npz")['arr_0']
    nbins = 25
    plt.hist(tr_snr, bins=nbins, color='k', histtype="step",
            lw=2, normed=True, alpha=0.3, label="Training Set")
    plt.hist(snr, bins=nbins, color='r', histtype="step",
            lw=2, normed=True, alpha=0.3, label="Validation Set")
    plt.legend()
    plt.xlabel("S/N", fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.ylabel("Normalized Count", fontsize=16)
    fig.tight_layout()
    plt.show()
    #plt.savefig("snr_dist.png")

def chisq_dist():
    fig = plt.figure(figsize=(6,4))
    ivar = np.load("%s/val_ivar_norm.npz" %DATA_DIR)['arr_0']
    npix = np.sum(ivar>0, axis=1)
    chisq = np.load("%s/val_chisq.npz" %DATA_DIR)['arr_0']
    redchisq = chisq/npix
    nbins = 25
    plt.hist(redchisq, bins=nbins, color='k', histtype="step",
            lw=2, normed=False, alpha=0.3, range=(0,3))
    plt.legend()
    plt.xlabel("Reduced $\chi^2$", fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.axvline(x=1.0, linestyle='--', c='k')
    fig.tight_layout()
    #plt.show()
    plt.savefig("chisq_dist.png")

def spectral_model():
    wl = np.load("../wl.npz")['arr_0']
    flux = np.load("../val_flux_norm.npz")['arr_0']
    ivar = np.load("../val_ivar_norm.npz")['arr_0']
    scat = np.load("../scatters.npz")['arr_0']
    coeffs = np.load("../coeffs.npz")['arr_0']
    pivots = np.load("../pivots.npz")['arr_0']
    cannon_label = np.load("../val_labels.npz")['arr_0']
    lvec_all = train_model._get_lvec(cannon_label, pivots)

    #xmin = min(wl)
    #xmax = max(wl)
    xmin = 8495
    xmax = 8585

    r_ymin = -0.1
    r_ymax = 0.1
    ymin = 0.25
    ymax = 1.1

    ii = 10
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
            wl, f, c='k', alpha=0.3, drawstyle='steps-mid', label="Data")
    ax_spectrum.plot(
            wl, model, c='r', alpha=0.3, label="The Cannon Model")
    ax_spectrum.fill_between(wl, model+err, model-err, alpha=0.1, color='r')
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
    #plt.show()
    #plt.savefig("model_spectrum_full.png")
    plt.savefig("model_spectrum.png")


if __name__=="__main__":
    chisq_dist()
