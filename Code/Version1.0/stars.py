"""
A Stars object is a set of stars: for example, a training set or a test set.

Parameters
---------
IDs: list of strings of length numstars
    list of filenames corresponding to each star's data file

spectra: 3D float array with shape (nstars, npixels, 3)
    spectra[:,:,0] = pixel wavelengths
    spectra[:,:,1] = flux array (spectrum)
    spectra[:,:,2] = flux err array (uncertainties in spectrum)

labels: None if unknown (like for the test set)
    otherwise, this is a float array of size (nstars, nlabels)
    this must be set for the training set

"""

import numpy as np

class Stars(object):
    
    def __init__(self, IDs, spectra, labels=None):
        self.IDs = IDs
        self.spectra = spectra 
        self.labels = labels 
  
    def set_IDs(self, IDs):
        self.IDs = IDs

    def set_spectra(self, spectra):
        self.spectra = spectra

    def set_label_values(self, label_values):
        label_names = self.get_label_names()
        self.labels = [label_names, label_values]

    def get_IDs(self):
        return self.IDs

    def get_spectra(self):
        return self.spectra

    def get_label_names(self):
        return self.labels[0]

    def get_label_values(self):
        return self.labels[1]

    def remove_stars(self, mask):
        """ Returns updated star object. True means remove star. """
        IDs = self.get_IDs()
        spectra = self.get_spectra()
        label_values = self.get_label_values()
        self.set_IDs(IDs[mask])
        self.set_spectra(spectra[mask])
        self.set_label_values(label_values[mask])
        print "stars removed"
        print mask
