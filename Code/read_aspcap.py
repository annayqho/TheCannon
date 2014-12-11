# This reads ASPCAP data and feeds it to the Spectra initialization class

from star import Star
import pyfits
import numpy as np
import os

def getSpectrum(fits_file):
    ' reads one .fits file and returns a spectrum array: [pixels, fluxes, errs] '
    file_in = pyfits.open(fits_file)
    fluxes = np.array(file_in[1].data)
    npixels = len(fluxes)
    flux_errs = np.array((file_in[2].data))
    start_wl = file_in[1].header['CRVAL1']
    diff_wl = file_in[1].header['CDELT1']
    val = diff_wl*(npixels) + start_wl
    wl_full_log = np.arange(start_wl,val, diff_wl)
    wl_full = [10**aval for aval in wl_full_log]
    pixels = np.array(wl_full) 
    spectrum = np.array([pixels, fluxes, flux_errs])
    return spectrum

def getTrainingSet():
    ' Return: filenames with corresponding training labels '
    ' In our case, there is one ages.txt file and one file with the rest of the params '
    file1 = "starsin_SFD_Pleiades.txt"
    file2 = "ages_2.txt"

    filenames = np.loadtxt(file1, usecols = (0,), dtype='string', unpack = 1)
    T_est,g_est,feh_est,T_A, g_A, feh_A = np.loadtxt(file1, usecols = (4,6,8,3,5,7), unpack =1)
    age_est = np.loadtxt(file2, usecols = (0,), unpack =1)
    label_values = np.array([T_est, g_est, feh_est, age_est])
    training_labels = label_values.T
    return filenames, training_labels

def getTestSet():
    ' Return: filenames ' 
    file1 = "starsin_SFD_Pleiades.txt"
    filenames = np.loadtxt(file1, usecols = (0,), dtype='string', unpack = 1)
    return filenames

def getStars(label_names=None):
    ' Returns an array of Star objects '
    
    if label_names is None:
        files = getTestSet()

    else:
        files, training_labels = getTrainingSet()

    nstars = len(files)
    stars = []

    for i in range(0, nstars):
        temp = files[i]
        # a hack to get the file location right...ugh
        temp1 = temp[1:]
        fits_file = '../Data/APOGEE_Data' + temp1

        nstar = np.where(files==temp)
        labels = [label_names, training_labels[nstar]]

        spectrum = getSpectrum(fits_file)
        star = Star(fits_file, spectrum, labels) # it has labels if it's a training star
        stars.append(star)

    return stars
