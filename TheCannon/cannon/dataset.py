from __future__ import (absolute_import, division, print_function)
import numpy as np
import matplotlib.pyplot as plt
from .helpers.triangle import corner
from cannon.helpers import Table
import sys
from .find_continuum_pixels import find_contpix

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
        wl, tr_fluxes, tr_ivars, tr_SNRs = self._load_spectra(training_dir)
        self.wl = wl
        self.tr_fluxes = tr_fluxes
        self.tr_ivars = tr_ivars
        self.tr_SNRs = tr_SNRs

        label_names, label_vals = self._load_reference_labels(label_file)
        self.tr_label_names = label_names
        self.tr_label_vals = label_vals

        wl, test_fluxes, test_ivars, test_SNRs = self._load_spectra(test_dir)
        self.test_fluxes = test_fluxes
        self.test_ivars = test_ivars
        self.test_SNRs = test_SNRs

    def _get_pixmask(self, *args, **kwags):
        raise NotImplemented('Derived classes need to implement this method')

    def _load_spectra(self, data_dir):
        raise NotImplemented('Derived classes need to implement this method')

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

    def reset_label_vals(self):
        self._label_vals = None

    def set_label_vals(self, vals):
        """ Set label vals from an array """
        self._label_vals = vals

    def set_label_names_tex(self, names):
        self.label_names_tex = names

    def get_plotting_labels(self):
        if self.label_names_tex is None:
            return self.label_names
        return self.label_names_tex
    
    def choose_labels(self, cols):
        """Updates the tr_label_names property

        Parameters
        ----------
        cols: list of column indices corresponding to which to keep
        """
        self.tr_label_names = []
        for k in cols:
            key = self.tr_label_vals.resolve_alias(k)
            if key not in self.tr_label_vals:
                raise KeyError('Attribute {0:s} not found'.format(key))
            else:
                self.tr_label_names.append(key)

    def label_triangle_plot(self, figname):
        """Make a triangle plot for the selected labels

        Parameters
        ----------
        figname: str
            if provided, save the figure into the given file

        labels: sequence
            if provided, use this sequence as text labels for each label
            dimension
        """
        data = np.array([self.tr_label_vals[k] for k in self.tr_label_names]).T
        labels = [r"$%s$" % l for l in self.get_plotting_labels()]
        print("Plotting every label against every other")
        fig = corner(data, labels=labels, show_titles=True,
                     title_args={"fontsize":12})
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

    def diagnostics_ref_labels(self, figname = "ref_labels_triangle.png"):
        """ Plot all training labels against each other. 
        
        Parameters
        ----------
        triangle_plot_name: (optional) string
            title of the saved triangle plot for reference labels
        """
        self.label_triangle_plot(figname)

    def find_continuum(self):
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
            contmask = find_contpix(self.wl, self.tr_fluxes, self.ivars)
        else:
            contmask = find_contpix_regions(self.wl, self.tr_fluxes, 
                                            self.tr_ivars, self.ranges)
        return contmask

    def continuum_normalize(self, contmask):
        """ Continuum normalize spectra

        For spectra split into regions, perform cont normalization
        separately for each region.
        """
        print("Continuum normalizing...")
        if self.ranges is None:
            print("assuming continuous spectra")
            norm_tr_fluxes, norm_tr_ivars = cont_norm(
                    self.tr_fluxes, self.tr_ivars, contmask)
            norm_test_fluxes, norm_test_ivars = cont_norm(
                    self.test_fluxes, self.test_ivars, contmask)
        else:
            norm_tr_fluxes, norm_tr_ivars = cont_norm_regions(
                    self.tr_fluxes, self.tr_ivars, contmask, self.ranges)
            norm_test_fluxes, norm_test_ivars = cont_norm_regions(
                    self.test_fluxes, self.test_ivars, contmask, self.ranges)

def dataset_postdiagnostics(reference_set, test_set,
                            triangle_plot_name = "survey_labels_triangle.png"):
    """ Run diagnostic tests on the test set after labels have been inferred.

    Tests result in the following output: one .txt file for each label listing
    all of the stars whose inferred labels lie >= 2 standard deviations outside
    the reference label space, a triangle plot showing all the survey labels 
    plotted against each other, and 1-to-1 plots for all of the labels showing
    how they compare to each other. 

    Parameters
    ----------
    reference_set: Dataset
        set used as training sample

    test_set: Dataset
        set for which labels are going to be inferred
    """
    # Find stars whose inferred labels lie outside the ref label space by 2-sig+
    label_names = reference_set.label_names
    nlabels = len(label_names)
    reference_labels = reference_set.label_vals
    test_labels = test_set.label_vals
    test_IDs = test_set.IDs
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
        print("flagged %s stars beyond 2-sig of reference labels" % sum(warning))
        print("Saved list %s" % filename)
    
    # Plot all survey labels against each other
    test_set.label_triangle_plot(triangle_plot_name)
    
    # 1-1 plots of all labels
    for i in range(nlabels):
        name = reference_set.get_plotting_labels()[i]
        orig = reference_labels[:,i]
        cannon = test_labels[:,i]
        low = np.minimum(min(orig), min(cannon))
        high = np.maximum(max(orig), max(cannon))
        fig, axarr = plt.subplots(2)
        ax1 = axarr[0]
        ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
        ax1.scatter(orig, cannon)
        ax1.legend()
        ax1.set_xlabel("Reference Value")
        ax1.set_ylabel("Cannon Output Value")
        ax1.set_title("1-1 Plot of Label " + r"$%s$" % name)
        ax2 = axarr[1]
        ax2.hist(cannon-orig)
        ax2.set_xlabel("Difference")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of Output Minus Ref Labels")
        figname = "1to1_label_%s.png" % i
        plt.savefig(figname)
        print("Diagnostic for label output vs. input")
        print("Saved fig %s" % figname)
        plt.close()


