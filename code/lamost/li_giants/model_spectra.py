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

DATA_DIR = "/Users/annaho/Data/LAMOST"
snr = np.load("%s/tr_snr.npz" %DATA_DIR)['arr_0']
chisq = np.load("%s/cannon_label_chisq.npz" %DATA_DIR)['arr_0']

wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
test_label = np.load("%s/all_cannon_labels.npz" %DATA_DIR)['arr_0']
ds.test_label_vals = test_label
tr_flux = np.load("%s/tr_flux.npz" %DATA_DIR)['arr_0']
tr_ivar = np.load("%s/tr_ivar.npz" %DATA_DIR)['arr_0']
ds.test_flux = tr_flux
ds.test_ivar = tr_ivar

m = model.CannonModel(2)
m.coeffs = np.load("%s/coeffs.npz" %DATA_DIR)['arr_0']
m.scatters = np.load("%s/scatters.npz" %DATA_DIR)['arr_0']
m.chisqs = np.load("%s/chisqs.npz" %DATA_DIR)['arr_0']
m.pivots = np.load("%s/pivots.npz" %DATA_DIR)['arr_0']

m.infer_spectra(ds)

choose = np.logical_and(snr > 100, chisq < 4000)
model_all = m.model_spectra[choose]
data = ds.test_flux[choose]
flux = ds.test_flux[choose]
ivar = ds.test_ivar[choose]
cannon_label = ds.test_label_vals

def spectral_model(ii):
    xmin = 6000
    xmax = 6800

    r_ymin = -0.05
    r_ymax = 0.05
    ymin = 0.6
    ymax = 1.15

    f = flux[ii,:]
    iv = ivar[ii,:]
    model = model_all[ii,:]

    # err = scat ^2 + uncertainty^2
    scat = m.scatters
    iv_tot = (iv/(scat**2 * iv + 1))
    err = np.ones(len(iv_tot))*1000
    err[iv_tot>0] = 1/iv_tot[iv_tot>0]**0.5

    print("X2 is: " + str(sum((f - model)**2 * iv_tot)))

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
    ax_residual.axvline(x=6707, c='r', linewidth=2, linestyle='--')
    ax_residual.axvline(x=6103, c='r', linewidth=2, linestyle='--')
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
    plt.axvline(x=6707, c='r', linewidth=2, linestyle='--')
    plt.axvline(x=6103, c='r', linewidth=2, linestyle='--')
    plt.show()
    #plt.savefig("model_spectrum_full.png")
    #plt.savefig("model_spectrum.png")


if __name__=="__main__":
    spectral_model(544)
