# This is the skeleton of the class used to extract spectra 
# (wavelengths, fluxes, flux errs) for training and test stars, 
# and training labels for the training stars. 

# Each survey or data type will inherit this class, ex. read_aspcap, 
# read_rave, read_uves. The user will then fill in methods as described below.

# The goal of this class is to create a Stars object. 
# Stars are represented by an array of continuum_normalized spectra 
# (shape = nstars, npixels, 3) and, if they are a training set, 
# labels (shape = nstars, nlabels).  

from stars import Stars
import pyfits
import numpy as np
import os

class ReadData():

    def __init__(self):
        pass

    ### Functions that need to be filled in by the user:

    def get_training_files(): 
       """ Returns: an array of length num_training_stars 
        consisting of the filenames corresponding to the training set 
        """ 
           
    def get_test_files():
        """ Returns: an array of length num_test_stars
        consisting of the filenames corresponding to the test set 
        """

    def get_wavelengths(filename):
        """ Input: name (string) of the data file containing the spectrum 
        Returns: np.array 
        that is the spectrum's x-axis (wavelengths in Angstroms)  
        """

    def get_fluxes(filename):
        """ Input: name (string) of the data file containing the spectrum 
        Returns: np.array 
        that is the y-axis (fluxes) of the spectrum
        """

    def get_fluxerrs(filename):
        """ Input: name (string) of the data file containing the spectrum
        Returns: np.array 
        that is the errors in the spectrum's y-axis (fluxes)
        """

    def get_training_labels(filename):
        """ Input: name (string) of the data file containing the labels
        Returns: a 2D np.array (size = num_training_stars, num_labels) consisting of all the training labels 
        """

    def set_stars_to_discard():
        ' Optional method, allows you to create a mask indicating which stars to throw out for whatever reason '
        ' Returns a boolean array of len(nstars) where True means "throw out" and false means "keep" '

    # Not sure what to do about this method
    def continuum_normalize(spectra):
        ' Reads the spectra object and applies some continuum normalization '
        ' Returns (1) a continuum-normalized version of the spectra object, shape (nstars, npixels, 3) and (2) the continua, shape (nstars, npixels)'

    ##### NOT changed by the user

    def get_spectra(files):
        ' Reads list of raw data files '
        ' Returns spectra array, shape (npixels, nstars, 3) '

    def get_stars(is_training, label_names):
        ' This method is NOT changed by the user '
        ' Constructs and returns a Stars object '
        if is_training:
            files = get_training_files()
            training_labels = get_training_labels()
        else:
            files = get_test_files()
            training_labels = None
        spectra = get_spectra(files)
        global nstars = spectra.shape[0]
        global npixels = spectra.shape[1]
        cont_norm_spectra = continuum_normalize(spectra)
        to_discard = discard_stars(training_labels, apogee_labels)
        stars = Stars(files, cont_norm_spectra, [label_names, training_labels])
        stars.removeStars(to_discard)
        return stars
