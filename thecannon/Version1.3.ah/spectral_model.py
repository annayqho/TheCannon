from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import os
import random
import math
import triangle
from dataset import Dataset
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colorbar

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
    residuals(cannon_set, test_set)

def split_array(array, num):
    avg = len(array) / float(num)
    out = []
    last = 0.0
    while last < len(array):
        out.append(array[int(last):int(last+avg)])
        last += avg
    return out

def overlay_spectra(cannon_set, test_set, model):
    coeffs_all, covs, scatters, chisqs, pivots, label_vector = model
    # Overplot original spectra with best-fit spectra
    res = test_set.fluxes-cannon_set.fluxes
    bad_ivar = np.std(res, axis=0) <= 1e-5
    os.system("mkdir SpectrumFits")
    print("Overplotting spectra for ten random stars")
    lambdas = test_set.lams
    npix = len(lambdas)
    nstars = cannon_set.fluxes.shape[0]
    pickstars = []
    for i in range(10):
        pickstars.append(random.randrange(0, nstars-1))
    for i in pickstars:
        print("Star %s" %i)
        ID = cannon_set.IDs[i]
        spec_orig = test_set.fluxes[i,:]
        bad_flux = np.logical_or(spec_orig == 0, spec_orig == 1) # unique to star
        bad = np.logical_or(bad_ivar, bad_flux)
        lambdas = np.ma.array(lambdas, mask=bad)
        npix = len(lambdas.compressed())
        spec_orig = np.ma.array(test_set.fluxes[i,:], mask=bad)
        spec_fit = np.ma.array(cannon_set.fluxes[i,:], mask=bad)
        ivars_orig = np.ma.array(test_set.ivars[i,:], mask=bad)
        ivars_fit = np.ma.array(cannon_set.ivars[i,:], mask=bad)
        red_chisq = np.sum(chisqs[:,i], axis=0)/(npix-coeffs_all.shape[1])
        red_chisq = np.round(red_chisq, 2)

        # 1-to-1 plot
        fig = plt.figure()
        im = plt.scatter(spec_orig, spec_fit, c=1/np.sqrt(ivars_orig), zorder=100)
        plt.errorbar(spec_orig, spec_fit, yerr=1/np.sqrt(ivars_fit), 
                     ecolor='k', fmt="none", alpha=0.3, zorder=0)
        textstr = "Red Chi Sq: %s" %red_chisq 
        cbar = plt.colorbar(im)
        cbar.set_label("Uncertainties on the Fluxes from the Original Spectrum")
        xlims = (np.ma.min(spec_orig), np.ma.max(spec_orig))
        ylims = (np.ma.min(spec_fit), np.ma.max(spec_fit))
        xlims = plt.xlim()
        ylims = plt.ylim()
        lims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
        plt.plot(lims, lims, 'k-', alpha=0.75)
        props = dict(boxstyle='round', facecolor='palevioletred', alpha=0.5)
        plt.annotate(textstr, fontsize=14, xy=(0.05, 0.90), 
                     xycoords='axes fraction')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel("Orig Fluxes")
        plt.ylabel("Fitted Fluxes")
        plt.title("1-to-1 Plot of Test vs. Best-Fit Spectrum")
        filename = "Star%s_1to1.png" %(i)
        print("Saved as %s" %filename)
        fig.savefig("SpectrumFits/"+filename)
        plt.close(fig)

        # Plot zoomed segments
        nseg = 100
        xmins = []
        xmaxs = []
        lams_seg = split_array(lams.data, nseg)
        for seg in lams_seg:
            xmins.append(seg[0])
            xmaxs.append(seg[-1])
        for j in range(10, 20):
            fig, axarr = plt.subplots(2, sharex=True, figsize=(15,10))
            plt.xlim(xmins[j], xmaxs[j])
            ax1 = axarr[0]
            ax1.scatter(lambdas, spec_orig, label="Orig Spec", zorder=100)
            ax1.errorbar(lambdas, spec_orig, yerr=1/np.sqrt(ivars_orig), 
                        alpha=0.7, fmt='ko', zorder=0)
            ax1.set_ylim(0.6, 1.2)
            ax1.scatter(lambdas, spec_fit, label="Cannon Spec", c='r', zorder=100)
            ax1.errorbar(lambdas, spec_fit, yerr=1/np.sqrt(ivars_fit), fmt='ro',
                         alpha=0.7, zorder=0)
            ax1.set_xlabel(r"Wavelength $\lambda (\AA)$")
            ax1.set_ylabel("Normalized flux")
            ax1.set_title("Spectrum Fit: %s" %ID)
            ax1.legend(loc='lower right', prop={'family':'serif', 'size':'small'})
            fig.subplots_adjust(right=0.8)
            
            ax2 = axarr[1]
            ax2.set_xlim(xmins[j], xmaxs[j])
            ax2.scatter(lambdas, spec_orig, label="Orig Spec", zorder=100)
            ax2.errorbar(lambdas, spec_orig, yerr=1/np.sqrt(ivars_orig), 
                        alpha=0.7, fmt='ko', zorder=0)
            ax2.set_xlabel(r"Wavelength $\lambda (\AA)$")
            ax2.set_ylabel("Normalized flux")
            ax2.set_title("Spectrum Fit, Zoomed")
            ax2.set_ylim(0.95, 1.05)
            ax2.scatter(lambdas, spec_fit, label="Cannon Spec", c='r', zorder=100)
            ax2.errorbar(lambdas, spec_fit, yerr=1/np.sqrt(ivars_fit), fmt='ro',
                         alpha=0.7, zorder=0)
            ax2.legend(loc='lower right', prop={'family':'serif', 'size':'small'})

            filename = "Star%s_section%s.png" %(i,j)
            print("Saved as %s" %filename)
            fig.savefig("SpectrumFits/"+filename)
            plt.close(fig)

def residuals(cannon_set, test_set):
    """ Stack spectrum fit residuals, sort by each label. Include histogram of
    the RMS at each pixel. 
    """
    print("Stacking spectrum fit residuals")
    res = test_set.fluxes-cannon_set.fluxes
    err = np.sqrt(1./test_set.ivars + 1./cannon_set.ivars)
    res_norm = res/err
    res_norm = np.ma.array(res_norm, 
            mask=(np.ones_like(res_norm)*(np.std(res_norm,axis=0)==0)))
    res_norm = np.ma.compress_cols(res_norm)
    for i in range(len(cannon_set.label_names)):
        label_name = cannon_set.label_names[i]
        print("Plotting residuals sorted by %s" %label_name)
        label_vals = cannon_set.label_vals[:,i]
        sorted_res = res_norm[np.argsort(label_vals)]
        mu = np.mean(sorted_res.flatten())
        sigma = np.std(sorted_res.flatten())
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.1
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, 0.1, height]
        rect_cbar = [left, bottom-0.2, 0.6, 0.06]
        plt.figure()
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        im = axScatter.imshow(sorted_res, cmap=plt.cm.bwr_r,
                interpolation="nearest", vmin=mu-3*sigma, vmax=mu+3*sigma,
                aspect='auto', origin='lower', extent=[0,
                    len(test_set.lams), min(label_vals), max(label_vals)])
        cax, kw = colorbar.make_axes(axScatter.axes, location='bottom')
        plt.colorbar(im, cax=cax, orientation='horizontal')
        axScatter.set_title("Spectral Residuals Sorted by " + r"$%s$" %label_name)
        axScatter.set_xlabel("Pixels")
        axScatter.set_ylabel(r"$%s$" %label_name)
        axHisty.hist(np.std(res_norm,axis=1), orientation='horizontal')
        axHisty.axhline(y=1, c='k', linewidth=3, label="y=1")
        axHisty.legend(bbox_to_anchor=(0.,0.8,1.,.102), 
                prop={'family':'serif', 'size':'small'})
        axHisty.text(1.0, 0.5, "Distribution of Stdev of Star Residuals", 
                verticalalignment='center', transform=axHisty.transAxes, rotation=270)
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
        filename = "residuals_sorted_by_label_%s.png" %i
        plt.savefig(filename)
        print("File saved as %s" %filename)
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
    print("saved %s" %filename)
    plt.close()
