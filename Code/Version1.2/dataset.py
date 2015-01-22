import numpy as np
import matplotlib.pyplot as plt
import triangle

"""Classes and methods for a Dataset of stars.

Provides the ability to initialize the dataset, modify it by adding or
removing spectra, changing label names, adding or removing labels.

Methods
-------
remove_stars

"""

class Dataset(object):
    """A class to represent a Dataset of stellar spectra and labels.
    
    Parameters
    ----------
    IDs: numpy ndarray, list
        Specify the names (or IDs, in whatever format) of the stars.
    lambdas: numpy ndarray
        Wavelength array corresponding to the pixels in the spectrum
    spectra: numpy ndarray
        spectra[:,:,0] = flux (spectrum)
        spectra[:,:,1] = flux error array
    labels: numpy ndarray, list, optional
        Training labels for training set, but None for test set
    
    Methods
    -------
    set_IDs
    set_spectra
    set_label_names
    set_label_vals
    choose_labels
    choose_spectra
    label_triangle_plot

    """

    def __init__(self, IDs, SNRs, lambdas, spectra, label_names, label_vals=None):
        self.IDs = IDs
        self.SNRs = SNRs
        self.lambdas = lambdas
        self.spectra = spectra 
        self.label_names = label_names
        self.label_vals = label_vals
  
    def set_IDs(self, IDs):
        self.IDs = IDs

    def set_lambdas(self, lambdas):
        self.lambdas = lambdas

    def set_spectra(self, spectra):
        self.spectra = spectra

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
        colmask[cols]=1
        new_label_vals = self.label_vals[:,colmask]
        self.set_label_names(new_label_names)
        self.set_label_vals(new_label_vals)

    def choose_spectra(self, mask):
        """Updates the ID, spectra, label_vals properties 

        Input: mask where 1 = keep, 0 = discard
        """
        self.set_IDs(self.IDs[mask])
        self.set_spectra(self.spectra[mask])
        self.set_label_vals(self.label_vals[mask])

    def label_triangle_plot(self, figname):
        """Plots every label against every other label"""
        fig = triangle.corner(self.label_vals, labels=self.label_names,
                show_titles=True, title_args = {"fontsize":12})
        fig.savefig(figname)
        print "Plotting every label against every other"
        print "Saved fig %s" %figname
        plt.close(fig)

def training_set_diagnostics(dataset):
    # Plot SNR distribution
    print "Diagnostic for SNR of training set"
    plt.hist(dataset.SNRs)
    plt.xscale('log')
    plt.title("Logspace Distribution of Formal SNR in the Training Set")
    plt.xlabel("log(Formal SNR)")
    plt.ylabel("Number of Objects")
    figname = "trainingset_SNRdist.png"
    plt.savefig(figname)
    plt.close()
    print "Saved fig %s" %figname
    
    # Plot training label distributions
    print "Diagnostic for coverage of training label space"
    for i in range(0, len(dataset.label_names)):
        name = dataset.label_names[i]
        vals = dataset.label_vals[:,i]
        plt.hist(vals)
        # Note: label names cannot have slashes 
        plt.title("Training Set Distribution of Label: %s" %name)
        plt.xlabel(name)
        plt.ylabel("Number of Objects")
        figname = "trainingset_labeldist_%s.png" %name
        plt.savefig(figname)
        print "Saved fig %s" %figname
        plt.close()
    
    # Plot all training labels against each other
    figname = "trainingset_labels_triangle.png"
    dataset.label_triangle_plot(figname)

def test_set_diagnostics(training_set, test_set):
    # 2-sigma check from training labels
    label_names = training_set.label_names
    nlabels = len(label_names)
    training_labels = training_set.label_vals
    test_labels = test_set.label_vals
    test_IDs = test_set.IDs
    mean = np.mean(training_labels, 0)
    stdev = np.std(training_labels, 0)
    lower = mean - 2 * stdev
    upper = mean + 2 * stdev
    for i in range(nlabels):
        label_name = label_names[i]
        test_vals = test_labels[:,i]
        warning = np.logical_or(test_vals < lower[i], test_vals > upper[i])
        flagged_stars = test_IDs[warning]
        filename = "flagged_stars_%s.txt" %label_name
        output = open(filename, 'w')
        for star in test_IDs[warning]:
            output.write(star + '\n')
        output.close()
        print "Training label %s" %label_name
        print "flagged %s stars beyond 2-sig of training labels" %sum(warning)
        print "Saved list %s" %filename
    
    # Plot all output labels against each other
    figname = "testset_labels_triangle.png"
    test_set.label_triangle_plot(figname)
    
    # 1-1 plots of all labels
    for i in range(nlabels):
        name = label_names[i]
        orig = training_labels[:,i]
        cannon = test_labels[:,i]
        plt.scatter(orig, cannon)
        plt.xlabel("Training Value")
        plt.ylabel("Cannon Output Value")
        plt.title("1-1 Plot of Label %s" %name)
        figname = "1to1_label%s.png" %name
        plt.savefig(figname)
        print "Diagnostic for label output vs. input"
        print "Saved fig %s" %figname
        plt.close()

def remove_stars(dataset, mask):
    """A method to remove a subset of stars from the Dataset. 
    
    Parameters
    ---------
    mask: numpy ndarray of booleans. True means "remove"
    """
    ntokeep = 1-np.sum(mask)
    return Dataset(dataset.IDs[mask], dataset.lambdas[mask], 
            dataset.spectra[mask], dataset.label_names, dataset.label_vals[mask])
    print "New dataset constructed with %s stars included" %ntokeep
