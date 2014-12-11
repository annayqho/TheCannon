# This reads ASPCAP data and feeds it to the Spectra initialization class

#from prep_data import Spectra
import pyfits
import numpy as np
import os

# the structure of an ASPCAP fits file: a[1].data is the flux, a[2].data is the error array. 

def get_spectrum(fits_file):
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

def get_spectra(fits_files):
    ' Takes in a list of .fits files and returns an array of spectra '
    nstars = len(fits_files)
    spectra = np.zeros(nstars)

    for nstar in range(0, nstars):
        fits_file = fits_files[nstar]
        spectrum = get_spectrum(fits_file)
        if nstar == 0:
            spectra = np.zeros((nstars,) + spectrum.shape)
        spectra[nstar] = spectrum
    
    return spectra

def get_training_labels(file1, file2):
    ' Read in the files that contain the training labels '
    ' Return: one array of label names, one array of label values '
    ' In our case, there is one ages.txt file and one file with the rest of the params '
    ' file1 = "starsin_SFD_Pleiades.txt" '
    ' file2 = "ages_2.txt" '

    T_est,g_est,feh_est,T_A, g_A, feh_A = np.loadtxt(fn, usecols = (4,6,8,3,5,7), unpack =1)
    age_est = np.loadtxt('ages_2.txt', usecols = (0,), unpack =1)
    label_names = ['Teff', 'logg', 'FeH', 'Age']
    label_values = [T_est, g_est, feh_est, age_est]
    return label_names, label_values

#### TESTING ####

dir = '/home/annaho/AnnaCannon/Code'
fitsfiles = [filename for filename in os.listdir(dir) if filename.endswith(".fits")]
spectra = get_spectra(fitsfiles)
print spectra.shape
