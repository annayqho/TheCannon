# make a list of SNRs corresponding to the sorted lamost stars

import os, pyfits, numpy as np

lamost = np.loadtxt("lamost_sorted_by_ra.txt", dtype=str)
#apogee = np.loadtxt("example_DR12/apogee_sorted_by_ra.txt", dtype=str)

lamost_sorted_by_snr = open("lamost_sorted_by_snr.txt", "w")

for i in range(len(lamost)):
    hdulist = pyfits.open('Data_All/%s' %lamost[i])
    flux = np.array(hdulist[0].data[0])
    ivar = np.array((hdulist[0].data[1]))
    snr = np.median(flux*ivar**0.5)
    lamost_sorted_by_snr.write(str(snr) + '\n')

lamost_sorted_by_snr.close()