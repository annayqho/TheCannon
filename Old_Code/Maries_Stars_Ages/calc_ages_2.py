# This script uses the coefficients obtained from the Cannon training data, plus the spectra of Marie's stars, and calculates ages in Gyr as well as log(age)

# Get the coefficients from the Cannon training data
# Gyr coefficients: /home/annaho/AnnaCannon/Code/Original_Code_newLitAges/coeffs_2nd_order.pickle
# log coefficients: /home/annaho/AnnaCannon/Code/Original_Code_logAges/coeffs_2nd_order.pickle

import scipy
import glob
import pylab
from scipy import interpolate
from scipy import ndimage
from scipy import optimize as opt
import pickle
import os
import fitspectra_ages_fast as f
import pyfits
import numpy as np

# Get the stellar spectra 

fn = 'apokasc_all_ages.txt'
T_est,g_est,feh_est,age_est = np.loadtxt(fn, usecols = (6,8,4,10), unpack =1) 
labels = ["teff", "logg", "feh", "age" ]

dir = '/home/annaho/AnnaCannon/Code/Maries_Data/'
file_list = []
for file in os.listdir(dir):
    if file.startswith("aspcapStar") and file.endswith(".fits"):
        file_list.append('%s%s' %(dir,file))

starlist = []

for filename in file_list:
    starname = filename.split('-')[2].split('.')[0]
    starlist.append(starname)

for jj, each in enumerate(file_list):
    numfiles = len(file_list)
    print "%s of %s" %(jj, numfiles)
    a = pyfits.open(each)
    b = pyfits.getheader(each)
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    if jj == 0:
        nmeta = len(labels) # number of parameters
        nlam = len(a[1].data) # number of pixels
    val = diff_wl*(nlam) + start_wl
    wl_full_log = np.arange(start_wl,val, diff_wl)
    ydata = (np.atleast_2d(a[1].data))[0]
    ydata_err = (np.atleast_2d(a[2].data))[0]
    ydata_flag = (np.atleast_2d(a[3].data))[0]
    assert len(ydata) == nlam
    wl_full = [10**aval for aval in wl_full_log]
    xdata= np.array(wl_full)
    ydata = np.array(ydata)
    ydata_err = np.array(ydata_err)
    sigma = (np.atleast_2d(a[2].data))[0]# /y1
    if jj == 0:
        npix = len(xdata) # the number of pixels
        dataall = np.zeros((npix, len(file_list), 3))
    if jj > 0:
        assert xdata[0] == dataall[0, 0, 0]

    dataall[:, jj, 0] = xdata
    dataall[:, jj, 1] = ydata
    dataall[:, jj, 2] = sigma

pixlist = np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1)

dataall_flat, continuum = f.continuum_normalize_tsch(dataall, pixlist, delta_lambda=50)

# dataall has shape (npixels, nstars, 3) where 3 represents the spectrum (x, y, yerr)

# In fitspectra.py, the routine infer_labels_nonlinear determines the labels for a new spectrum 

fn_pickle = '/home/annaho/AnnaCannon/Code/litAges_fast/coeffs_2nd_order.pickle'
#fn_pickle = '/home/annaho/AnnaCannon/Code/Original_Code_logAges/coeffs_2nd_order.pickle'

f.infer_labels_nonlinear(fn_pickle, dataall_flat, 'self_2nd_order_tags.pickle', -10.950, 10.99)




