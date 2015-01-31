"""Classes and methods for a Dataset of stars.

Provides the ability to initialize the dataset, modify it by adding or
removing spectra, changing label names, adding or removing labels.

Methods
-------
remove_stars

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
from .helpers.triangle import corner


class Dataset(object):
    """A class to represent a Dataset of stellar spectra and labels.

    Attributes
    ----------
    IDs: numpy ndarray, list
        Specify the names (or IDs, in whatever format) of the stars.
    lambdas: numpy ndarray
        Wavelength array corresponding to the pixels in the spectrum
    spectra: numpy ndarray
        spectra[:,:,0] = flux (spectrum)
        spectra[:,:,1] = flux error array
    labels: numpy ndarray, list, optional
        Reference labels for reference set, but None for test set
    """

    def __init__(self, IDs, SNRs, lams, fluxes, ivars, label_names, label_vals=None):
        self.IDs = IDs
        self.SNRs = SNRs
        self.lams = lams
        self.fluxes = fluxes
        self.ivars = ivars
        self.label_names = label_names
        self.label_vals = label_vals

    def set_IDs(self, IDs):
        self.IDs = IDs

    def set_lambdas(self, lams):
        self.lams = lams

    def set_fluxes(self, fluxes):
        self.fluxes = fluxes

    def set_ivars(self, ivars):
        self.ivars = ivars

    def set_label_names(self, label_names):
        self.label_names = label_names

    def set_label_vals(self, label_vals):
        self.label_vals = label_vals

    def choose_labels(self, cols):
        """Updates the label_names and label_vals properties

        Input: list of column indices corresponding to which to keep
        """
        new_label_names = [self.label_names[i] for i in cols]
        colmask = np.zeros(len(self.label_names), dtype=bool)
        colmask[cols] = 1
        new_label_vals = self.label_vals[:, colmask]
        self.set_label_names(new_label_names)
        self.set_label_vals(new_label_vals)

    def choose_objects(self, mask):
        """Updates the ID, spectra, label_vals properties

        Input: mask where 1 = keep, 0 = discard
        """
        self.set_IDs(self.IDs[mask])
        self.set_fluxes(self.fluxes[mask])
        self.set_ivars(self.ivars[mask])
        self.set_label_vals(self.label_vals[mask])

    def label_triangle_plot(self, figname):
        """Plots every label against every other label"""
        texlabels = []
        for label in self.label_names:
            texlabels.append(r"$%s$" % label)
        fig = corner(self.label_vals, labels=texlabels, show_titles=True,
                     title_args={"fontsize":12})
        fig.savefig(figname)
        print("Plotting every label against every other")
        print("Saved fig %s" % figname)
        plt.close(fig)


def dataset_prediagnostics(reference_set, test_set):
    # Plot SNR distributions
    print("Diagnostic for SNRs of reference and survey stars")
    plt.hist(reference_set.SNRs, alpha=0.5, label="Ref Stars")
    plt.hist(test_set.SNRs, alpha=0.5, label="Survey Stars")
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.title("SNR Comparison Between Reference & Test Stars")
    plt.xlabel("log(Formal SNR)")
    plt.ylabel("Number of Objects")
    figname = "SNRdist.png"
    plt.savefig(figname)
    plt.close()
    print("Saved fig %s" % figname)

    # Plot all reference labels against each other
    figname = "reference_labels_triangle.png"
    reference_set.label_triangle_plot(figname)


def dataset_postdiagnostics(reference_set, test_set):
    # 2-sigma check from reference labels
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
        # assigned but never used
        # flagged_stars = test_IDs[warning]
        filename = "flagged_stars_%s.txt" % i
        output = open(filename, 'w')
        for star in test_IDs[warning]:
            output.write(star + '\n')
        output.close()
        print("Reference label %s" % label_name)
        print("flagged %s stars beyond 2-sig of reference labels" % sum(warning))
        print("Saved list %s" % filename)
    # Plot all survey labels against each other
    figname = "survey_labels_triangle.png"
    test_set.label_triangle_plot(figname)
    # 1-1 plots of all labels
    for i in range(nlabels):
        name = label_names[i]
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
