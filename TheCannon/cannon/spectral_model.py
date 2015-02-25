from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import os
import random
from .helpers.triangle import corner
import matplotlib.pyplot as plt
from matplotlib import colorbar
from copy import deepcopy

LARGE = 100.
SMALL = 1. / LARGE

def draw_spectra(model, dataset):
    """ Generate a set of spectra given the model and test set labels 

    Parameters
    ----------
    model:

    test_set:

    Returns
    -------
    cannon_set
    """
    
    coeffs_all, covs, scatters, red_chisqs, pivots, label_vector = model
    nstars = len(dataset.test_SNRs)
    cannon_fluxes = np.zeros(dataset.test_fluxes.shape)
    cannon_ivars = np.zeros(dataset.test_ivars.shape)
    for i in range(nstars):
        x = label_vector[:,i,:]
        spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
        cannon_fluxes[i,:] = spec_fit
        bad = dataset.test_ivars[i,:] == SMALL
        cannon_ivars[i,:][~bad] = 1. / scatters[~bad] ** 2
    cannon_set = deepcopy(dataset)
    cannon_set.test_fluxes = cannon_fluxes
    cannon_set.test_ivars = cannon_ivars

    return cannon_set


def diagnostics(cannon_set, dataset, model):
    """ Run diagnostics on the fitted spectra

    Parameters
    ----------
    cannon_set:

    test_set:

    model:
    """
    overlay_spectra(cannon_set, dataset, model)
    residuals(cannon_set, dataset)


def triangle_pixels(cannon_set):
    """ Make a triangle plot for the pixel fluxes, to see correlation
    
    Parameters
    ----------
    cannon_set
    """
    
    fluxes = cannon_set.test_fluxes
    corner(fluxes)


def overlay_spectra(cannon_set, dataset, model):
    """ Run a series of diagnostics on the fitted spectra 

    Parameters
    ----------

    cannon_set:

    test_set:

    model:
    """
    
    coeffs_all, covs, scatters, chisqs, pivots, label_vector = model
    # Overplot original spectra with best-fit spectra
    res = dataset.test_fluxes-cannon_set.test_fluxes
    bad_ivar = np.std(res, axis=0) <= 1e-5
    os.system("mkdir SpectrumFits")
    print("Overplotting spectra for ten random stars")
    lambdas = dataset.wl
    npix = len(lambdas)
    nstars = cannon_set.test_fluxes.shape[0]
    pickstars = []
    for i in range(10):
        pickstars.append(random.randrange(0, nstars-1))
    for i in pickstars:
        print("Star %s" % i)
        ID = cannon_set.test_IDs[i]
        spec_orig = dataset.test_fluxes[i,:]
        bad_flux = (spec_orig == 0) | (spec_orig == 1)  # unique to star
        bad = (bad_ivar | bad_flux)
        lambdas = np.ma.array(lambdas, mask=bad, dtype=float)
        npix = len(lambdas.compressed())
        spec_orig = np.ma.array(dataset.test_fluxes[i,:], mask=bad)
        spec_fit = np.ma.array(cannon_set.test_fluxes[i,:], mask=bad)
        ivars_orig = np.ma.array(dataset.test_ivars[i,:], mask=bad)
        ivars_fit = np.ma.array(cannon_set.test_ivars[i,:], mask=bad)
        #fig, axarr = plt.subplots(2)
        #ax1 = axarr[0]
        #ax1.scatter(lambdas, spec_orig, label="Orig Spec",
        #            c=1. / np.sqrt(ivars_orig))
        red_chisq = np.sum(chisqs[:,i], axis=0) / (npix - coeffs_all.shape[1])
        red_chisq = np.round(red_chisq, 2)
        fig,axarr = plt.subplots(2)
        ax1 = axarr[0]
        im = ax1.scatter(lambdas, spec_orig, label="Orig Spec",
                         c=1 / np.sqrt(ivars_orig))
        ax1.scatter(lambdas, spec_fit, label="Cannon Spec", c='r')
        ax1.errorbar(lambdas, spec_fit, yerr=1/np.sqrt(ivars_fit), fmt='ro')
        ax1.set_xlabel(r"Wavelength $\lambda (\AA)$")
        ax1.set_ylabel("Normalized flux")
        ax1.set_title("Spectrum Fit: %s" % ID)
        ax1.set_title("Spectrum Fit")
        ax1.legend(loc='lower center', fancybox=True, shadow=True)
        ax2 = axarr[1]
        ax2.scatter(spec_orig, spec_fit, c=1/np.sqrt(ivars_orig))
        ax2.errorbar(spec_orig, spec_fit, yerr=1 / np.sqrt(ivars_fit),
                     ecolor='k', fmt="none")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(
                im, cax=cbar_ax,
                label="Uncertainties on the Fluxes from the Original Spectrum")
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        lims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
        ax2.plot(lims, lims, 'k-', alpha=0.75)
        textstr = "Red Chi Sq: %s" % red_chisq
        props = dict(boxstyle='round', facecolor='palevioletred', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.set_xlabel("Orig Fluxes")
        ax2.set_ylabel("Fitted Fluxes")
        filename = "Star%s.png" % i
        print("Saved as %s" % filename)
        fig.savefig("SpectrumFits/" + filename)
        plt.close(fig)


def residuals(cannon_set, dataset):
    """ Stack spectrum fit residuals, sort by each label. Include histogram of
    the RMS at each pixel.

    Parameters
    ----------
    cannon_set:

    test_set:
    """
    print("Stacking spectrum fit residuals")
    res = dataset.test_fluxes - cannon_set.test_fluxes
    bad = dataset.test_ivars == SMALL
    err = np.zeros(len(dataset.test_ivars))
    err = np.sqrt(1. / dataset.test_ivars + 1. / cannon_set.test_ivars)
    res_norm = res / err
    res_norm = np.ma.array(res_norm,
                           mask=(np.ones_like(res_norm) *
                                 (np.std(res_norm,axis=0) == 0)))
    res_norm = np.ma.compress_cols(res_norm)

    for i in range(len(cannon_set.get_plotting_labels())):
        label_name = cannon_set.get_plotting_labels()[i]
        print("Plotting residuals sorted by %s" % label_name)
        label_vals = cannon_set.tr_label_vals[:,i]
        sorted_res = res_norm[np.argsort(label_vals)]
        mu = np.mean(sorted_res.flatten())
        sigma = np.std(sorted_res.flatten())
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.1
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, 0.1, height]
        plt.figure()
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        im = axScatter.imshow(sorted_res, cmap=plt.cm.bwr_r,
                              interpolation="nearest", vmin=mu - 3. * sigma,
                              vmax=mu + 3. * sigma, aspect='auto',
                              origin='lower', extent=[0, len(dataset.wl),
                                                      min(label_vals),
                                                      max(label_vals)])
        cax, kw = colorbar.make_axes(axScatter.axes, location='bottom')
        plt.colorbar(im, cax=cax, orientation='horizontal')
        axScatter.set_title(
                r"Spectral Residuals Sorted by ${0:s}$".format(label_name))
        axScatter.set_xlabel("Pixels")
        axScatter.set_ylabel(r"$%s$" % label_name)
        axHisty.hist(np.std(res_norm,axis=1), orientation='horizontal')
        axHisty.axhline(y=1, c='k', linewidth=3, label="y=1")
        axHisty.legend(bbox_to_anchor=(0., 0.8, 1., .102),
                       prop={'family':'serif', 'size':'small'})
        axHisty.text(1.0, 0.5, "Distribution of Stdev of Star Residuals",
                     verticalalignment='center', transform=axHisty.transAxes,
                     rotation=270)
        axHisty.set_ylabel("Standard Deviation")
        start, end = axHisty.get_xlim()
        axHisty.xaxis.set_ticks(np.linspace(start, end, 3))
        axHisty.set_xlabel("Number of Stars")
        axHisty.xaxis.set_label_position("top")
        axHistx.hist(np.std(res_norm, axis=0))
        axHistx.axvline(x=1, c='k', linewidth=3, label="x=1")
        axHistx.set_title("Distribution of Stdev of Pixel Residuals")
        axHistx.set_xlabel("Standard Deviation")
        axHistx.set_ylabel("Number of Pixels")
        start, end = axHistx.get_ylim()
        axHistx.yaxis.set_ticks(np.linspace(start, end, 3))
        axHistx.legend()
        filename = "residuals_sorted_by_label_%s.png" % i
        plt.savefig(filename)
        print("File saved as %s" % filename)
        plt.close()

    # Auto-correlation of mean residuals
    print("Plotting Auto-Correlation of Mean Residuals")
    mean_res = res_norm.mean(axis=0)
    autocorr = np.correlate(mean_res, mean_res, mode="full")
    pkwidth = int(len(autocorr)/2-np.argmin(autocorr))
    xmin = int(len(autocorr)/2)-pkwidth
    xmax = int(len(autocorr)/2)+pkwidth
    zoom_x = np.linspace(xmin, xmax, len(autocorr[xmin:xmax]))
    fig, axarr = plt.subplots(2)
    axarr[0].plot(autocorr)
    axarr[0].set_title("Autocorrelation of Mean Spectral Residual")
    axarr[0].set_xlabel("Lag (# Pixels)")
    axarr[0].set_ylabel("Autocorrelation")
    axarr[1].plot(zoom_x, autocorr[xmin:xmax])
    axarr[1].set_title("Central Peak, Zoomed")
    axarr[1].set_xlabel("Lag (# Pixels)")
    axarr[1].set_ylabel("Autocorrelation")
    filename = "residuals_autocorr.png"
    plt.savefig(filename)
    print("saved %s" % filename)
    plt.close()
