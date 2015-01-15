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
    plt.close()
    print "Diagnostic for SNR of training set"
    print "Saved fig %s" %figname
    # Plot training label distributions
    for i in range(0, len(dataset.label_names)):
        name = dataset.label_names[i]
        vals = dataset.label_values[:,i]
        plt.hist(vals)
        # Note: label names cannot have slashes 
        plt.title("Training Set Distribution of Label: %s" %name)
        figname = "trainingset_labeldist_%s.png" %name
        plt.savefig(figname)
        print "Diagnostic for coverage of training label space"
        print "Saved fig %s" %figname
        plt.close()
    # Plot all training labels against each other
    fig = triangle.corner(dataset.label_values, labels=dataset.label_names, 
            show_titles=True, title_args = {"fontsize": 12})
    figname = "trainingset_labels_triangle.png"
    fig.savefig(figname)
    print "Diagnostic for plotting every training label against every other"
    print "Saved fig %s" %figname

def test_set_diagnostics(training_set, test_set):
    label_names = training_set.label_names
    nlabels = len(label_names)
    training_labels = training_set.label_values
    test_IDs = test_set.IDs
    mean = np.mean(training_labels, 0)
    stdev = np.std(training_labels, 0)
    lower = mean - 2 * stdev
    upper = mean + 2 * stdev
    for i in range(nlabels):
        label_name = label_names[i]
        test_vals = test_set.label_values[:,i]
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
