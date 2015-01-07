"""
This file is part of The Cannon analysis project.
Original version copyright 2014 Melissa Ness.
This version copyright 2014 Anna Ho.

Before you start, make sure you have the following files in the directory:
    pixtest4.txt
    starsin_SFD_Pleiades.txt ("training_labels_file")
        stars = rows, 1st col = ID, other cols = labels, first row = # then headers
    and in the dir_fits directory, you need the .fits files
        num of .fits files should correspond to the number of stars
        and the files must include the IDs that constitute the first column of training_labels_file
# Ideas for improvement:
    -- Test the files to make sure that they are structured correctly, ex. whether the number of rows and columns is all consistent, whether the headers correspond to the right kinds of numbers (check to see range and average of Feh, z.B.)
    -- Test whether the stellar IDs all match up with each other

Helpful links --
documentation for aspcapStar files: http://data.sdss3.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/ASPCAP_VERS/RESULTS_VERS/LOCATION_ID/aspcapStar.html

"""

import glob
import pickle
import numpy as np
import os
import pyfits

# defined by the user
training_labels_file = "newcode_apokasc_all_ages.txt" 
    # Note: this file should have stars as rows, labels as columns
    # Note: the first row should have # then label headers
labels = ['Teff', 'seismic_logg', 'FeH', 'max_age']
    # Note: these should correspond to the headers in the first row of training_labels
dir_fits = '/home/annaho/AnnaCannon/Code/Maries_Data' # the directory that holds the .fits files
samplefits = "aspcapStar-v304-2M19212264+4901500.fits"
sampleID = "J18501318+4139450"
ID_start = 1 # index where the number starts

# defined within the Cannon
normed_training_data_file = "normed_data.pickle"
nlabels = len(labels)
fitsfiles = [filename for filename in os.listdir(dir_fits) if filename.endswith(".fits")]
nstars = len(fitsfiles)

# Reads in the data file that contains the training data, normalizes it, returns it
def get_normalized_training_data_tsch(pixlist):
    if glob.glob(normed_training_data_file):
        print "Warning: normed_data.pickle already exists"
        file_in = open(normed_training_data_file, 'r')
        dataall, metaall, labels, Ametaall, cluster_name, ids = pickle.load(file_in)
        file_in.close()
        return dataall, metaall, labels, Ametaall, cluster_name, ids
    print "normed_data.pickle not found, making a new one..."
    
    # Read in the labels of the training data
    file_in = open(training_labels_file, 'r')
    headers = filter(None, file_in.readline().strip('\n').split(" "))
    file_in.close()

    cols = [] # corresponding to labels
    for label in labels:
        col = headers.index(label) - 1 # because of the # in the header line
        cols.append(col)
    
    IDs = np.loadtxt(training_labels_file, usecols = (0,), dtype=np.str, unpack = 1)
    training_labels = np.loadtxt(training_labels_file, usecols = cols, unpack = 1)

    ### Now that the stars are identified, read in the spectrum from the corresponding .fits file
    for i,fitsfile in fitsfiles:
        print "%s of %s" %(i, nstars)
        a = pyfits.open(fitsfile)

        # How the hell do you come up with xdata? 

        ydata = (np.atleast_2d(a[1].data))[0]
        ydata_err = (np.atleast_2d(a[2].data))[0]
        a.close()

if __name__ == "__main__":
    # I don't know what this pixlist actually is, but I should say something about it here in the comments.
    # pixlist = np.loadtxt("pixtest4.txt", usecols = (0,), unpack = 1)

    # retrieve the data for the training set
    print "retrieving normalized training set data"
    dataall, metaall, labels, Ametaall, cluster_name, ids = get_normalized_training_data_tsch(pixlist)

    
