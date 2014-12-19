# Like star.py, except each object is an array of stars
# Initialize all of the data: spectra and training labels

import numpy as np

class Stars(object):
    ' Common base class for all stars, regardless of survey, training and test '
    def __init__(self, IDs, spectra, labels=None):
        self.IDs = IDs
        self.spectra = spectra # size (nstars, npixels, 3)
        self.labels = labels # size (nstars, nlabels)
   
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
        ' Mask has True for remove star, False for keep star '
        ' Returns updated star object '
        IDs = self.get_IDs()
        spectra = self.get_spectra()
        label_values = self.get_label_values()
        self.set_IDs(IDs[mask])
        self.set_spectra(spectra[mask])
        self.set_label_values(label_values[mask])

    # You can imagine there being some kind of def sanity_check() here, where the self-consistency of all of the input data is tested, ex. that the length of the arrays match, etc
