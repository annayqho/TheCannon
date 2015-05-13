from __future__ import (absolute_import, division, print_function)
import numpy as np
import matplotlib.pyplot as plt
from .helpers.triangle import corner
from .helpers import Table
import sys
from .find_continuum_pixels import find_contpix, find_contpix_regions
from .continuum_normalization import fit_cont, fit_cont_regions, cont_norm, cont_norm_regions, cont_norm_q

PY3 = sys.version_info[0] > 2

if PY3:
    basestring = (str, bytes)
else:
    basestring = (str, unicode)


class Dataset(object):
    """A class to represent a Dataset of stellar spectra and labels.

    Initialize this object after performing the munging necessary 
    for making data "Cannonizable."
    """

    def __init__(self, wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar):
        print("Loading dataset")
        print("This may take a while...")
        self.wl = wl
        self.tr_flux = tr_flux
        self.tr_ivar = tr_ivar
        self.tr_label = tr_label
        self.test_flux = test_flux
        self.test_ivar = test_ivar
        self.ranges = None
        
        # calculate SNR
        self.tr_SNR = np.array(
                [self._SNR(*s) for s in zip(tr_flux, tr_ivar)])
        self.test_SNR = np.array(
                [self._SNR(*s) for s in zip(test_flux, test_ivar)])


    def _SNR(self, flux, ivar):
        """ Calculate the SNR of a spectrum, ignoring bad pixels

        Parameters
        ----------
        flux: numpy ndarray
            pixel intensities
        ivar: numpy ndarray
            inverse variances corresponding to flux

        Returns
        -------
        SNR: float
        """
        bad = ivar == 0
        flux = np.ma.array(flux, mask=bad)
        ivar = np.ma.array(ivar, mask=bad)
        SNR = float(np.ma.median(flux*ivar**0.5))
        return SNR  


    def set_label_names(self, names):
        """ Set the label names for plotting

        Parameters
        ---------
        names: array, list
            the names of the labels
        """
        self._label_names = names


    def get_plotting_labels(self):
        """ Return the labels set for plotting

        Returns
        -------
        the label names
        """
        if self._label_names is None:
            print("No label names yet!")
            return None
        else:
            return self._label_names
 

    def diagnostics_SNR(self, figname = "SNRdist.png"): 
        """ Plot SNR distributions of ref and test objects

        Parameters
        ----------
        figname: (optional) string
            title of the saved SNR diagnostic plot
        """
        print("Diagnostic for SNRs of reference and survey objects")
        data = self.tr_SNR
        plt.hist(data, bins=np.sqrt(len(data)), alpha=0.5, color='b',
                label="Ref Objects")
        data = self.test_SNR
        plt.hist(data, bins=np.sqrt(len(data)), alpha=0.5, facecolor='r', 
                label="Survey Objects")
        plt.legend(loc='upper right')
        #plt.xscale('log')
        plt.title("SNR Comparison Between Reference \& Survey Objects")
        #plt.xlabel("log(Formal SNR)")
        plt.xlabel("Formal SNR")
        plt.ylabel("Number of Objects")
        plt.savefig(figname)
        plt.close()
        print("Saved fig %s" %figname)

    
    def diagnostics_ref_labels(self, figname="ref_labels_triangle.png"):
        """ Plot all training labels against each other. 
        
        Parameters
        ----------
        figname: (optional) string
            title of the saved triangle plot for reference labels
        """
        self.label_triangle_plot(self.tr_label, figname)


    def label_triangle_plot(self, label_vals, figname):
        """Make a triangle plot for the selected labels

        Parameters
        ----------
        label_vals: numpy ndarray 
            values of the labels
        figname: str
            if provided, save the figure into the given file
        """
        labels = [r"$%s$" % l for l in self.get_plotting_labels()]
        print("Plotting every label against every other")
        fig = corner(label_vals, labels=labels, show_titles=True,
                     title_args={"fontsize":12})
        fig.savefig(figname)
        print("Saved fig %s" % figname)
        plt.close(fig)


    def make_contmask(self, fluxes, ivars, frac):
        """ Use spectra to find and return continuum pixels

        For spectra split into regions, performs cont pix identification
        separately for each region.
        
        Parameters
        ----------
        fluxes: numpy ndarray
            pixel intensities

        ivars: numpy ndarray
            inverse variances for pixel fluxes

        frac: float
            the fraction of pixels that should be identified as continuum

        Returns
        -------
        contmask: boolean mask of length npixels
            True indicates that the pixel is continuum
        """
        print("Finding continuum pixels...")
        if self.ranges is None:
            print("assuming continuous spectra")
            contmask = find_contpix(self.wl, fluxes, ivars, frac)
        else:
            print("taking spectra in %s regions" %len(self.ranges))
            contmask = find_contpix_regions(
                    self.wl, fluxes, ivars, frac, self.ranges)
        print("%s pixels returned as continuum" %sum(contmask))
        return contmask


    def set_continuum(self, contmask):
        """ Set the contmask attribute 

        Parameters
        ---------
        contmask: boolean numpy ndarray
            True corresponds to continuum pixels
        """
        self.contmask = contmask


    def diagnostics_contmask(self, figname='contpix.png'):
        """ Make and save diagnostic figure about the continuum pixels

        Parameters
        ----------
        (optional) figname: str
            name of the figure to be saved
        """
        contmask = self.contmask
        f_bar = np.zeros(len(self.wl))
        sigma_f = np.zeros(len(self.wl))
        for wl in range(0,len(self.wl)):
            flux = self.tr_flux[:,wl]
            ivar = self.tr_ivar[:,wl]
            f_bar[wl] = np.median(flux[ivar>0])
            sigma_f[wl] = np.sqrt(np.var(flux[ivar>0]))
        bad = np.var(self.tr_ivar, axis=0) == 0
        f_bar = np.ma.array(f_bar, mask=bad)
        sigma_f = np.ma.array(sigma_f, mask=bad)
        plt.plot(self.wl, f_bar, alpha=0.7)
        plt.fill_between(self.wl, (f_bar+sigma_f), (f_bar-sigma_f), alpha=0.2)
        plt.scatter(self.wl[contmask], f_bar[contmask], c='r', label="Cont Pix")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Median Flux Across Training Objects")
        plt.legend()
        plt.title("Continuum Pix Found by The Cannon")
        plt.savefig(figname)
        print("Saving fig %s" %figname)


    def fit_continuum(self, deg, ffunc):
        """ Fit a continuum to the continuum pixels

        Parameters
        ---------
        deg: int
            degree of the fitting function
        ffunc: str
            type of fitting function, sinusoid or chebyshev
        """
        if self.ranges == None:
            tr_cont = fit_cont(
                    self.tr_flux, self.tr_ivar, self.contmask, deg, ffunc)
            test_cont = fit_cont(
                    self.test_flux, self.test_ivar, self.contmask, deg, ffunc)
        else:
            tr_cont = fit_cont_regions(self.tr_flux, self.tr_ivar, 
                                       self.contmask, deg, self.ranges, ffunc)
            test_cont = fit_cont_regions(self.test_flux, self.test_ivar,
                                         self.contmask, deg, self.ranges, ffunc)
        return tr_cont, test_cont


    def continuum_normalize_training_q(self, q, delta_lambda):
        """ Continuum normalize the training set using a running quantile

        Parameters
        ---------
        q: float
            the quantile cut
        delta_lambda: float
            the width of the pixel range used to calculate the median
        """
        print("Continuum normalizing the tr set using running quantile...")
        return cont_norm_q(
            self.wl, self.tr_flux, self.tr_ivar, q=q, delta_lambda=delta_lambda)


    def continuum_normalize_f(self, cont):
        """ Continuum normalize spectra by dividing the spectrum by the continuum 

        For spectra split into regions, perform cont normalization
        separately for each region.

        Parameters
        ----------
        cont: numpy ndarray
           the continuum 

        Returns
        -------
        norm_tr_flux: numpy ndarray
            the normalized training objects' pixel intensities
        norm_tr_ivar: numpy ndarray
            the rescaled training objects' pixel inverse variances
        norm_test_flux: numpy ndarray
            the normalized test objects' pixel intensities
        norm_test_ivar: numpy ndarray
            the rescaled test objects' pixel inverse variances
        """
        tr_cont, test_cont = cont
        if self.ranges is None:
            print("assuming continuous spectra")
            norm_tr_flux, norm_tr_ivar = cont_norm(
                    self.tr_flux, self.tr_ivar, tr_cont)
            norm_test_flux, norm_test_ivar = cont_norm(
                    self.test_flux, self.test_ivar, test_cont)
        else:
            print("taking spectra in %s regions" %(len(self.ranges)))
            norm_tr_flux, norm_tr_ivar = cont_norm_regions(
                    self.tr_flux, self.tr_ivar, tr_cont, self.ranges)
            norm_test_flux, norm_test_ivar = cont_norm_regions(
                    self.test_flux, self.test_ivar, test_cont, self.ranges)
        return norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar


    def diagnostics_test_step_flagstars(self):
        """ list all stars whose inferred labels lie >= 2 standard deviations
        outside the reference label space """
        label_names = self.get_plotting_labels()
        nlabels = len(label_names)
        reference_labels = self.tr_label
        test_labels = self.test_label_vals
        test_IDs = np.array(self.test_IDs)
        mean = np.mean(reference_labels, 0)
        stdev = np.std(reference_labels, 0)
        lower = mean - 2 * stdev
        upper = mean + 2 * stdev
        for i in range(nlabels):
            label_name = label_names[i]
            test_vals = test_labels[:,i]
            warning = np.logical_or(test_vals < lower[i], test_vals > upper[i])
            filename = "flagged_stars_%s.txt" % i
            with open(filename, 'w') as output:
                for star in test_IDs[warning]:
                    output.write('{0:s}\n'.format(star))
        print("Reference label %s" % label_name)
        print("flagged %s stars beyond 2-sig of ref labels" % sum(warning))
        print("Saved list %s" % filename)


    def diagnostics_survey_labels(self, figname="survey_labels_triangle.png"):
        """ make a triangle plot for all the survey labels 
        Parameters
        ----------
        (optional) figname: str
            name of plot to be saved
        """  
        figname="survey_labels_triangle.png"
        self.label_triangle_plot(self.test_label_vals, figname)
   
   
    def diagnostics_1to1(self):
        """ 1to1 plots of survey against training labels, color-coded by test SNR
        
        assumes your training set and test set are identical"""
        snr = self.test_SNR
        for i in range(nlabels):
            name = self.get_plotting_labels()[i]
            orig = reference_labels[:,i]
            cannon = test_labels[:,i]
            # calculate bias and scatter
            scatter = np.round(np.std(orig-cannon),3)
            bias  = np.round(np.mean(orig-cannon),3)

            low = np.minimum(min(orig), min(cannon))
            high = np.maximum(max(orig), max(cannon))

            fig = plt.figure(figsize=(10,6))
            gs = gridspec.GridSpec(1,2,width_ratios=[2,1], wspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
            ax1.set_xlim(low, high)
            ax1.set_ylim(low, high)
            ax1.legend(fontsize=14)
            pl = ax1.scatter(orig, cannon, marker='x', c=snr,
                    vmin=50, vmax=200, alpha=0.7)
            cb = plt.colorbar(pl, ax=ax1, orientation='horizontal')
            cb.set_label('SNR from Test Set', fontsize=12)
            textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                    fontsize=14, verticalalignment='top')
            ax1.tick_params(axis='x', labelsize=14)
            ax1.tick_params(axis='y', labelsize=14)
            ax1.set_xlabel("Reference Value", fontsize=14)
            ax1.set_ylabel("Cannon Test Value", fontsize=14)
            ax1.set_title("1-1 Plot of Label " + r"$%s$" % name)
            diff = cannon-orig
            npoints = len(diff)
            mu = mean(diff)
            sig = std(diff)
            ax2.hist(diff, range=[-3*sig,3*sig], color='k', bins=np.sqrt(npoints),
                    orientation='horizontal', alpha=0.3, histtype='stepfilled')
            ax2.tick_params(axis='x', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)
            ax2.set_xlabel("Count", fontsize=14)
            ax2.set_ylabel("Difference", fontsize=14)
            ax2.axhline(y=0, c='k', lw=3, label='Difference=0')
            ax2.set_title("Training Versus Test Labels for $%s$" %name,
                    fontsize=14)
            ax2.legend(fontsize=14)
            figname = "1to1_label_%s.png" % i
            plt.savefig(figname)
            print("Diagnostic for label output vs. input")
            print("Saved fig %s" % figname)
            plt.close()


    def set_test_label_vals(self, vals):
        """ Set label vals from an array 

        Parameters
        ----------
        vals: value of the labels
        """
        self.test_label_vals = vals
