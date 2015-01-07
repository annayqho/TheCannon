' This is the skeleton of the file that the user will modify. '
' The purpose is to read in the raw spectra and labels and create a Stars object, which consists of continuum-normalized spectra (wavelengths, fluxes, flux errs) as well as training labels. '

from stars import Stars
import pyfits
import numpy as np
import os

npixels = 0
nstars = 0

def __init__(self):
    pass

def get_training_files():
    ' Returns: array of filenames of length (ntrainingstars) '

def get_test_files():
    ' Returns: array of filenames of length (nteststars) '

def get_spectra(files):
    ' Reads list of raw data files '
    ' Returns spectra array, shape (npixels, nstars, 3) '

def continuum_normalize(spectra):
    ' Reads the spectra object and applies some continuum normalization '
    ' Returns (1) a continuum-normalized version of the spectra object, shape (nstars, npixels, 3) and (2) the continua, shape (nstars, npixels)'

def get_training_labels():
    ' Returns: 2D array of size (ntrainingstars, nlabels) '

def discard_stars():
    ' Optional method, allows you to create a mask indicating which stars to throw out for whatever reason '
    ' Returns a boolean array of len(nstars) where True means "throw out" and false means "keep" '
    
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
