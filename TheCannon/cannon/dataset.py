from __future__ import (absolute_import, division, print_function)
import numpy as np
import matplotlib.pyplot as plt
from .helpers.triangle import corner
from cannon.helpers import Table
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

    Framework for performing the munging necessary for making data "Cannonizable."
    Each survey will have its own implementation of the following: how data is 
    retrieved, how bad pixels are identified. Packages all of this information
    into rectangular blocks.
    """

    def __init__(self, training_dir, test_dir, label_file):
        IDs, wl, fluxes, ivars, SNRs = self._load_spectra(training_dir)
        self.tr_IDs = IDs
        self.wl = wl
        self.tr_fluxes = fluxes
        self.tr_ivars = ivars
        self.tr_SNRs = SNRs

        label_names, label_data = self._load_reference_labels(label_file)
        self.label_names = label_names
        self.tr_label_data = label_data
        self.test_label_vals = None
        self.reset_label_vals()
        
        IDs, wl, fluxes, ivars, SNRs = self._load_spectra(test_dir)
        self.test_IDs = IDs
        self.test_fluxes = fluxes
        self.test_ivars = ivars
        self.test_SNRs = SNRs

        self.contmask = None

    def _get_pixmask(self, *args, **kwags):
        raise NotImplemented('Derived classes need to implement this method')

    def _load_spectra(self, data_dir):
        raise NotImplemented('Derived classes need to implement this method')

    def reset_label_vals(self):
        self._tr_label_vals = None
    
    def set_test_label_vals(self, vals):
        """ Set label vals from an array """
        self.test_label_vals = vals

    def set_label_names_tex(self, names):
        self.label_names_tex = names

    @property
    def tr_label_vals(self):
        """ return the array of labels [Nsamples x Ndim] """
        if self._tr_label_vals is None:
            return np.array([self.tr_label_data[k] for 
                            k in self.label_names]).T
        else:
            return self._tr_label_vals

    def _load_reference_labels(self, label_file):
        """Extracts training labels from file.

        Assumes that first row is # then label names, that first column is # 
        then the filenames, that the remaining values are floats and that 
        user wants all of the labels. User can pick specific labels later.

        Returns
        -------
        data['id']: 
        label_names: list of label names
        data: np ndarray of size (nstars, nlabels)
            label values
        """
        print("Loading reference labels from file %s" %label_file)
        data = Table(label_file)
        data.sort('id')
        label_names = data.keys()
        nlabels = len(label_names)

        print("Loaded stellar IDs, format: %s" % data['id'][0])
        print("Loaded %d labels:" % nlabels)
        print(label_names)
        return label_names, data


    def get_plotting_labels(self):
        if self.label_names_tex is None:
            return self.label_names
        return self.label_names_tex
    
    def choose_labels(self, cols):
        """Updates the label_names property

        Parameters
        ----------
        cols: list of column indices corresponding to which to keep
        """
        self.label_names = []
        for k in cols:
            key = self.tr_label_data.resolve_alias(k)
            if key not in self.tr_label_data:
                raise KeyError('Attribute {0:s} not found'.format(key))
            else:
                self.label_names.append(key)

    def label_triangle_plot(self, label_vals, figname):
        """Make a triangle plot for the selected labels

        Parameters
        ----------
        figname: str
            if provided, save the figure into the given file

        labels: sequence
            if provided, use this sequence as text labels for each label
            dimension
        """
        labels = [r"$%s$" % l for l in self.get_plotting_labels()]
        print("Plotting every label against every other")
        fig = corner(label_vals, labels=labels, show_titles=True,
                     title_args={"fontsize":12})
        print("figname: %s" %figname)
        fig.savefig(figname)
        print("Saved fig %s" % figname)
        plt.close(fig)

    def diagnostics_SNR(self, figname = "SNRdist.png"): 
        """ Plot SNR distributions of ref and test objects

        Parameters
        ----------
        SNR_plot_name: (optional) string
            title of the saved SNR diagnostic plot
        """
        print("Diagnostic for SNRs of reference and survey stars")
        plt.hist(self.tr_SNRs, alpha=0.5, label="Ref Stars")
        plt.hist(self.test_SNRs, alpha=0.5, label="Survey Stars")
        plt.legend(loc='upper right')
        plt.xscale('log')
        plt.title("SNR Comparison Between Reference & Test Stars")
        plt.xlabel("log(Formal SNR)")
        plt.ylabel("Number of Objects")
        plt.savefig(figname)
        plt.close()
        print("Saved fig %s" %figname)

    def diagnostics_ref_labels(self, figname="ref_labels_triangle.png"):
        """ Plot all training labels against each other. 
        
        Parameters
        ----------
        triangle_plot_name: (optional) string
            title of the saved triangle plot for reference labels
        """
        label_vals = np.array([self.tr_label_data[k] 
                              for k in self.label_names]).T
        self.label_triangle_plot(label_vals, figname)

    def find_gaps(self, fluxes):
        # Gaps: regions where median(flux) == 0., and var(flux) == 0.
        gaps = np.logical_and(np.median(fluxes, axis=0) == 0, 
                              np.std(fluxes, axis=0) == 0)    
        return gaps

    def find_continuum(self, f_cut=0.003, sig_cut=0.003):
        """ Use training spectra to find and return continuum pixels

        For spectra split into regions, performs cont pix identification
        separately for each region.
        
        Returns
        -------
        contmask: boolean mask of length npixels
            True indicates that the pixel is continuum
        """
        print("Finding continuum pixels...")
        if self.ranges is None:
            print("assuming continuous spectra")
            contmask = find_contpix(f_cut, sig_cut, self.wl, 
                    self.tr_fluxes, self.tr_ivars)
        else:
            contmask = find_contpix_regions(self.wl, self.tr_fluxes, 
                                            self.tr_ivars, self.ranges)
        print("%s pixels returned as continuum" %sum(contmask))
        self.contmask = contmask

    def set_continuum(self, contmask):
        self.contmask = contmask

    def fit_continuum(self, deg=3):
        if self.ranges == None:
            tr_cont = fit_cont(
                    self.tr_fluxes, self.tr_ivars, self.contmask, deg)
            test_cont = fit_cont(
                    self.test_fluxes, self.test_ivars, self.contmask, deg)
        else:
            tr_cont = fit_cont_regions(self.tr_fluxes, self.tr_ivars, 
                                       self.contmask, deg, self.ranges)
            test_cont = fit_cont_regions(self.tr_fluxes, self.tr_ivars,
                                         self.contmask, deg, self.ranges)
            
        return tr_cont, test_cont


    def continuum_normalize_q(self, q, delta_lambda):
        """ Continuum normalize spectra using a running quantile."""
        print("Continuum normalizing using running percentile...")
        norm_tr_fluxes, norm_tr_ivars = cont_norm_q(
                self.wl, self.tr_fluxes, self.tr_ivars, 
                q=q, delta_lambda=delta_lambda)
        norm_test_fluxes, norm_test_ivars = cont_norm_q(
                self.wl, self.test_fluxes, self.test_ivars, 
                q=q, delta_lambda=delta_lambda)
        return norm_tr_fluxes, norm_tr_ivars, norm_test_fluxes, norm_test_ivars


    def continuum_normalize_f(self):
        """ Continuum normalize spectra by fitting a function to continuum pix 

        For spectra split into regions, perform cont normalization
        separately for each region.
        """
        if self.ranges is None:
            print("assuming continuous spectra")
            norm_tr_fluxes, norm_tr_ivars = cont_norm(
                    self.tr_fluxes, self.tr_ivars, dataset.contmask)
            norm_test_fluxes, norm_test_ivars = cont_norm(
                    self.test_fluxes, self.test_ivars, dataset.contmask)
        else:
            print("taking spectra in %s regions" %(len(self.ranges)))
            norm_tr_fluxes, norm_tr_ivars, cont = cont_norm_regions(
                    self.tr_fluxes, self.tr_ivars, 
                    dataset.contmask, self.ranges)
            norm_test_fluxes, norm_test_ivars, cont = cont_norm_regions(
                    self.test_fluxes, self.test_ivars, 
                    dataset.contmask, self.ranges)
        return norm_tr_fluxes, norm_tr_ivars, norm_test_fluxes, norm_test_ivars


    def dataset_postdiagnostics(self, figname="survey_labels_triangle.png"):
        """ Run diagnostic tests on the test set after labels have been inferred.

        Tests result in the following output: one .txt file for each label 
        listing all of the stars whose inferred labels lie >= 2 standard 
        deviations outside the reference label space, a triangle plot showing 
        all the survey labels plotted against each other, and 1-to-1 plots 
        for all of the labels showing how they compare to each other. 

        Parameters
        ----------
        """
        # Find stars whose inferred labels lie >2-sig outside ref label space
        label_names = self.label_names
        nlabels = len(label_names)
        reference_labels = self.tr_label_vals
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
    
        # Plot all survey labels against each other
        figname="survey_labels_triangle.png"
        self.label_triangle_plot(self.test_label_vals, figname)
    
        # 1-1 plots of all labels
        for i in range(nlabels):
            name = self.get_plotting_labels()[i]
            orig = reference_labels[:,i]
            cannon = test_labels[:,i]
            # calculate bias and scatter
            scatter = np.round(np.std(orig-cannon),3)
            bias  = np.round(np.mean(orig-cannon),3)
            low = np.minimum(min(orig), min(cannon))
            high = np.maximum(max(orig), max(cannon))
            fig, axarr = plt.subplots(2)
            ax1 = axarr[0]
            ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
            ax1.scatter(orig, cannon)
            textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top')
            ax1.set_xlabel("Reference Value")
            ax1.set_ylabel("Cannon Output Value")
            ax1.set_title("1-1 Plot of Label " + r"$%s$" % name)
            ax2 = axarr[1]
            ax2.hist(cannon-orig, range=[-0.5,0.5])
            ax2.set_xlabel("Difference")
            ax2.set_ylabel("Count")
            ax2.set_title("Histogram of Output Minus Ref Labels")
            figname = "1to1_label_%s.png" % i
            plt.savefig(figname)
            print("Diagnostic for label output vs. input")
            print("Saved fig %s" % figname)
            plt.close()


