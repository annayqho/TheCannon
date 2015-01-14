import numpy as np
import matplotlib.pyplot as plt

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
    spectra: numpy ndarray
        spectra[:,:,0] = wavelengths, pixel values
        spectra[:,:,1] = flux (spectrum)
        spectra[:,:,2] = flux error array
    labels: numpy ndarray, list, optional
        Training labels for training set, but None for test set
    
    Methods
    -------
    set_IDs
    set_spectra
    set_label_names
    set_label_values

    """

    def __init__(self, IDs, SNRs, spectra, label_names, label_values=None):
        self.IDs = IDs
        self.SNRs = SNRs
        self.spectra = spectra 
        self.label_names = label_names
        self.label_values = label_values
        if label_values is not None:
            self.plot_SNRs()
            self.plot_labelspace()
  
    def set_IDs(self, IDs):
        self.IDs = IDs

    def set_spectra(self, spectra):
        self.spectra = spectra

    def set_label_names(self, label_names):
        self.label_names = label_names

    def set_label_values(self, label_values):
        self.label_values = label_values

def training_set_diagnostics(dataset):
    # Plot SNR distribution
    plt.hist(dataset.SNRs)
    plt.title("Distribution of SNR in the Training Set")
    figname = "trainingset_SNRdist.png"
    plt.savefig(figname)
    print "Diagnostic for SNR of training set"
    print "Saved fig %s" %figname
    # Plot training label distribution
    for i in range(0, len(dataset.label_names)):
        name = dataset.label_names[i]
        vals = dataset.label_values[:,i]
        plt.hist(vals)
        # Note: label names cannot have slashes 
        plt.title("Distribution of Label: %s" %name)
        figname = "labeldist_%s.png" %name
        plt.savefig(figname)
        print "Diagnostic for coverage of training label space"
        print "Saved fig %s" %figname

def remove_stars(dataset, mask):
    """A method to remove a subset of stars from the Dataset. 
    
    Parameters
    ---------
    mask: numpy ndarray of booleans. True means "remove"
    """
    ntokeep = 1-np.sum(mask)
    return Dataset(dataset.IDs[mask], dataset.spectra[mask],
            dataset.label_names, dataset.label_values[mask])
    print "New dataset constructed with %s stars included" %ntokeep
