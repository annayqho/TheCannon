import numpy as np
import glob
from lamost import load_spectra
allfiles = np.array(glob.glob("example_LAMOST/Data_All/*fits")) 
# we want just the file names 
allfiles = np.char.lstrip(allfiles, 'example_LAMOST/Data_All/') 
dir_dat = "example_LAMOST/Data_All"
ID, wl, flux, ivar = load_spectra(dir_dat, allfiles)
npix = np.array([np.count_nonzero(ivar[jj,:]) for jj in range(0,11057)])
good_frac = npix/3626. 
SNR_raw = flux * ivar**0.5
bad = SNR_raw == 0
SNR_raw = np.ma.array(SNR_raw, mask=bad)
SNR = np.ma.median(SNR_raw, axis=1)

# we want to have at least 94% of pixels, and SNR of at least 100 
good = np.logical_and(good_frac > 0.94, SNR>100) 
tr_files = ID[good] #945 spectra 
outputf = open("tr_files.txt", "w")
for tr_file in tr_files: 
    outputf.write(tr_file + '\n')
outputf.close()
