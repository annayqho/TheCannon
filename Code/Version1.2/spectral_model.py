import numpy as np
import os
import random
from dataset import Dataset
import matplotlib.pyplot as plt

def draw_spectra(model, test_set):
    coeffs_all, covs, scatters, red_chisqs, pivots, label_vector = model
    nstars = len(test_set.IDs)
    cannon_fluxes = np.zeros(test_set.fluxes.shape)
    cannon_ivars = np.zeros(test_set.ivars.shape)
    for i in range(nstars):
        x = label_vector[:,i,:]
        spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
        cannon_fluxes[i,:] = spec_fit
        cannon_ivars[i,:] = 1. / scatters**2
    cannon_set = Dataset(IDs=test_set.IDs, SNRs=test_set.SNRs, 
            lams = test_set.lams, fluxes=cannon_fluxes, ivars = cannon_ivars,
            label_names = test_set.label_names, 
            label_vals = test_set.label_vals)
    return cannon_set

def diagnostics(cannon_set, test_set, model):
    overlay_spectra(cannon_set, test_set, model)
    residuals(cannon_set, test_set, model)

def overlay_spectra(cannon_set, test_set, model):
    coeffs_all, covs, scatters, red_chisqs, pivots, label_vector = model
    # Overplot original spectra with best-fit spectra
    os.system("mkdir SpectrumFits")
    print "Overplotting spectra for ten random stars"
    lambdas = test_set.lams
    nstars = cannon_set.fluxes.shape[0]
    pickstars = []
    for i in range(10):
        pickstars.append(random.randrange(0, nstars-1))
    for i in pickstars:
        print "Star %s" %i
        ID = cannon_set.IDs[i]
        spec_orig = test_set.fluxes[i,:]
        spec_fit = cannon_set.fluxes[i,:]
        ivars = test_set.ivars[i,:]
        sigma2 = 1. / ivars
        err_orig = np.sqrt(sigma2)
        err_fit = np.sqrt(scatters**2)
        #err_fit = np.sqrt(sigma2 + scatters**2)
        bad_flux = np.logical_or(spec_orig == 0, spec_orig == 1)
        bad_ivar = ivars == 0.
        bad = np.logical_or(bad_flux, bad_ivar)
        keep = np.invert(bad)
        chisq = np.round(red_chisqs[i], 2)
        fig,axarr = plt.subplots(2)
        ax1 = axarr[0]
        im = ax1.scatter(lambdas[keep], spec_orig[keep], 
                label="Orig Spec", c=err_orig[keep])
        ax1.scatter(lambdas[keep], spec_fit[keep], label="Cannon Spec", c='r')
        ax1.errorbar(lambdas[keep], spec_fit[keep], yerr=err_fit[keep], fmt='ro')
        ax1.set_xlabel(r"Wavelength $\lambda (\AA)$")
        ax1.set_ylabel("Normalized flux")
        ax1.set_title("Spectrum Fit: %s" %ID)
        ax1.legend(loc='lower center', fancybox=True, shadow=True)
        ax2 = axarr[1]
        ax2.scatter(spec_orig[keep], spec_fit[keep], c=err_orig[keep])
        ax2.errorbar(spec_orig[keep], spec_fit[keep], yerr=err_fit[keep], 
                ecolor='k', fmt="none")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        im.set_label("Uncertainties on the Fluxes from the Original Spectrum")
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        lims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
        ax2.plot(lims, lims, 'k-', alpha=0.75)
        textstr = "Red Chi Sq: %s" %chisq
        props = dict(boxstyle='round', facecolor='palevioletred', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.set_xlabel("Orig Fluxes")
        ax2.set_ylabel("Fitted Fluxes")
        filename = "Star%s.png" %i
        print "Saved as %s" %filename
        fig.savefig("SpectrumFits/"+filename)
        plt.close(fig)

def residuals(cannon_set, test_set, model):
    coeffs_all, covs, scatters, red_chisqs, pivots, label_vector = model
    print "Stacking spectrum fit residuals"
    res = test_set.fluxes-cannon_set.fluxes
    spec_fit = cannon_set.fluxes
    err = np.sqrt(1./test_set.ivars + scatters**2)
    res_norm = res/err
    for i in range(len(cannon_set.label_names)):
        label_name = cannon_set.label_names[i]
        print "Plotting residuals sorted by %s" %label_name
        label_vals = cannon_set.label_vals[:,i]
        sorted_res = res_norm[np.argsort(label_vals)]
        mu = np.mean(sorted_res.flatten())
        sigma = np.std(sorted_res.flatten())
        #lim = np.maximum(np.abs(sorted_res.max()), np.abs(sorted_res.min()))
        plt.imshow(sorted_res, cmap=plt.cm.bwr_r,
                interpolation="nearest", vmin=mu-3*sigma, vmax=mu+3*sigma,
                aspect='auto',origin='lower')
        plt.title("Spectral Residuals Sorted by " + r"$%s$" %label_name)
        plt.xlabel("Pixels")
        plt.ylabel("Stars")
        plt.colorbar()
        filename = "residuals_sorted_by_label_%s.png" %i
        plt.savefig(filename)
        print "File saved as %s" %filename
        plt.close()
    print "Plotting Auto-Correlation of Mean Residuals"
    mean_res = res_norm.mean(axis=0)
    autocorr = np.correlate(mean_res, mean_res, mode="full")
    plt.plot(autocorr)
    plt.title("Autocorrelation of Mean Spectral Residual")
    plt.xlabel("k")
    plt.ylabel("r_k")
    filename = "residuals_autocorr.png" 
    plt.savefig(filename)
    print "saved %s" %filename
    plt.close()
