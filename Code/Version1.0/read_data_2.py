"""
This is the skeleton of the class used to establish the training set and test set for input into The Cannon.

The user does not interact with this class. Instead, the user writes a child class that inherits from this class, and fills in methods that indicate how spectra (wavelengths, fluxes, flux errs) and training labels (for the training set) should be extracted from the raw data files, the conditions (if any) that determine which stars they want to discard.  

The last method in this class, set_star_set, is not changed by the user. It creates the training_set and test_set using the methods defined by the user. Note: this method should probably be moved to a different file, but for now it lives here.
"""

from stars import Stars
import pyfits
import numpy as np
import os

class ReadData():

    def __init__(self):
        pass

    ### The following functions need to be filled in by the user:

    def get_spectra(files):
        """
        Extracts spectra (wavelengths, fluxes, fluxerrs) from raw data files
        
        Input: a list of data file names of length nstars 
        Returns: a 3D float array of shape (nstars, npixels, 3) 
        with spectra[:,:,0] = pixel wavelengths
        spectra[:,:,1] = flux values
        spectra[:,:,2] = flux err values
        """

    def continuum_normalize(spectra):
        """
        Continuum-normalizes the spectra. 

        Input: spectra array, which is a 3D float array w/ shape (nstars,npixels,3)
        Returns:    3D continuum-normalized spectra (shape=nstars,npixels,3)
                    2D continuum array (shape=nstars,npixels)
        """

    def get_training_files():
        """ 
        Establishes which files correspond to the training set data. 

        Returns: an array of filenames of length ntrainingstars 
        """

    def get_test_files():
        """
        Establishes which files correspond to the test set data.
        
        Returns an array of filenames of length nteststars 
        """

    def get_training_labels(filename):
        """
        Extracts training labels from file

        Input: name (string) of the data file containing the labels
        Returns: a 2D np.array (size = num_training_stars, num_labels) 
        consisting of all the training labels 
        """

    def set_stars_to_discard():
        """
        Enables the user to create a mask indicating which stars to throw out.
        Sample criteria: logg within a specific range
        
        Returns: a boolean array of len(nstars) where True means "discard"
        """

    ##### The following function is NOT changed by the user.
    ##### This should probably live in a different file. 

    def set_star_set(is_training, label_names):
        """ 
        Constructs and returns a Stars object, which consists of a set of spectra and (if it's the training set) training labels. Uses the methods defined above by the user.

        Input:  is_training boolean (True or False), True if it's the training set
                label_names, a list of strings, ex. ['FeH', 'logg']
        Returns: a Stars object, corresponding to ex. the training set
        """
        if is_training:
            files = get_training_files()
            training_labels = get_training_labels()
        else:
            files = get_test_files()
            training_labels = None
        spectra = get_spectra(files)
        cont_norm_spectra = continuum_normalize(spectra)
        to_discard = set_stars_to_discard()
        stars = Stars(files, cont_norm_spectra, [label_names, training_labels])
        stars.removeStars(to_discard)
        return stars
