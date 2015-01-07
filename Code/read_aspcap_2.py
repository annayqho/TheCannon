# This is the ASPCAP implementation of the read_file class. 

from read_data_2 import ReadData
from stars import Stars
import pyfits
import numpy as np
import os

class ReadASPCAP(ReadData):

    def __init__(self):
        super(ReadASPCAP, self).__init__()

    def get_spectra(files):
        ' Reads file list and returns spectra array, shape (npixels, nstars, 3) '
        for jj,fits_file in enumerate(files):
            file_in = pyfits.open(fits_file)
            fluxes = np.array(file_in[1].data)
            if jj == 0:
                global nstars 
                nstars = len(files)
                global npixels 
                npixels = len(fluxes)
                spectra = np.zeros((nstars, npixels, 3))
            flux_errs = np.array((file_in[2].data))
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl*(npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10**aval for aval in wl_full_log]
            pixels = np.array(wl_full) 
            spectra[jj, :, 0] = pixels
            spectra[jj, :, 1] = fluxes
            spectra[jj, :, 2] = flux_errs
        return spectra

    def find_gaps(spectra):

        # - Find the gaps in the spectrum, assume a series of zeros with len(>=50 pix)
        # - There are a few stars for which the gap is not quite the same.  
        allstarts = []
        allends = []
    
        for jj in range(38, 50):
            loczero = np.where(spectra[jj,:,1] == 0)[0]
            starts = []
            ends = []
            startind = 0
            endind = 1
            while endind < len(loczero):
                if (loczero[endind] - loczero[endind-1] == 1):
                    endind = endind + 1
                else:
                    if endind - startind >= 200:
                        starts.append(loczero[startind])
                        ends.append(loczero[endind-1])
                    startind = endind
                    endind = startind + 1
            starts.append(loczero[startind])
            ends.append(loczero[endind-1])
            starts = np.array(starts)
            ends = np.array(ends)
            allstarts.append(starts)
            allends.append(ends)

    def continuum_normalize(spectra):
        ' Fit 2nd order Chebyshev polynomial to each segment of spectrum and divide by it '
        continua = np.zeros((nstars, npixels))
        normalized_spectra = np.ones((nstars, npixels, 3))
   
        # pixlist is a list of "true" continuum pixels, det in this case by the Cannon
        pixlist = list(np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1))

        # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels
        ## The regions with flux were found to be: [321, 3242] [3647, 6047], [6411, 8305]
        ## With edge cuts: [371, 3192], [3697, 5997], [6461, 8255]
        ## Corresponding to: [15218, 15743] [15931, 16367] [16549, 16887] 

        ranges = [[371,3192], [3697,5997], [6461,8255]]
        LARGE = 200.

        # Fit the Chebyshev polynomial and continuum-normalize each region separately
        for jj in range(nstars):
            # Mask unphysical pixels
            bad1 = np.invert(np.logical_and(np.isfinite(spectra[jj,:,1]),  np.isfinite(spectra[jj,:,2])))
            bad = bad1 | (spectra[jj,:,2] <= 0)
            spectra[jj,:,1][bad] = 0.
            spectra[jj,:,2][bad] = np.Inf

            var_array = 100**2*np.ones(npixels)
            var_array[pixlist] = 0.000
            ivar = 1. / ((spectra[jj, :, 2] ** 2) + var_array)
            ivar = (np.ma.masked_invalid(ivar)).filled(0)
            for i in range(len(ranges)):
                start, stop = ranges[i][0], ranges[i][1]
                spectrum = spectra[jj,start:stop,:]
                ivar1 = ivar[start:stop]
                fit = np.polynomial.chebyshev.Chebyshev.fit(x=spectrum[:,0], y=spectrum[:,1], w=ivar1, deg=3)
                continua[jj,start:stop] = fit(spectrum[:,0])
                normalized_fluxes = spectrum[:,1]/fit(spectra[0,start:stop,0])
                bad = np.invert(np.isfinite(normalized_fluxes))
                normalized_fluxes[bad] = 1.
                normalized_flux_errs = spectrum[:,2]/fit(spectra[0,start:stop,0])
                bad = np.logical_or(np.invert(np.isfinite(normalized_flux_errs)), normalized_flux_errs <= 0)
                normalized_flux_errs[bad] = LARGE
                normalized_spectra[jj,:,0] = spectra[jj,:,0]
                normalized_spectra[jj,start:stop,1] = normalized_fluxes 
                normalized_spectra[jj,start:stop,2] = normalized_flux_errs
        
            # Another unphysical pixel check
            bad = spectra[jj,:,2] > LARGE
            normalized_spectra[jj,np.logical_or(bad, bad1),1] = 1.
            normalized_spectra[jj,np.logical_or(bad, bad1),2] = LARGE

        return normalized_spectra, continua

    # These two functions are currently exactly the same because of the nature of the APOGEE set...this is for a sanity check where the test set == the training set
    def get_training_files():
        ' Return: filenames array of length (ntrainingstars) '
        readin = "starsin_SFD_Pleiades.txt"
        filenames = np.loadtxt(readin, usecols = (0,), dtype='string', unpack = 1)
        filenames1 = [] # for some reason if I try to replace the element, it gets rid of the '.fits' at the end...
        for i in range(0, len(filenames)): # incorporate file location info
            filename = '../Data/APOGEE_Data' + filenames[i][1:] 
            filenames1.append(filename)
        return np.array(filenames1)

    def get_test_files():
        ' Return: filenames array of length (nteststars) ' 
        readin = "starsin_SFD_Pleiades.txt"
        filenames = np.loadtxt(readin, usecols = (0,), dtype='string', unpack = 1)
        filenames1 = [] # for some reason if I try to replace the element, it gets rid
        for i in range(0, len(filenames)):
            filename = '../Data/APOGEE_Data' + filenames[i][1:] 
            filenames1.append(filename)
        return np.array(filenames1)

    def get_training_labels():
        ' Return: 2D array of size (ntrainingstars, nlabels) '
        input1 = "starsin_SFD_Pleiades.txt"
        input2 = "ages_2.txt"
        T_est,g_est,feh_est,T_A, g_A, feh_A = np.loadtxt(input1, usecols = (4,6,8,3,5,7), unpack =1)
        age_est = np.loadtxt(input2, usecols = (0,), unpack =1)
        training_labels = np.array([T_est, g_est, feh_est, age_est])
        apogee_labels = np.array([T_A, g_A, feh_A])
        return training_labels.T, apogee_labels.T

    def discard_stars(training_labels, apogee_labels):
        ' Return: A mask telling you which stars to throw out '
        diff_t = np.abs(apogee_labels[:,0]-training_labels[:,0]) # temp difference
        logg_cut = 100.
        diff_t_cut = 600.
        bad = np.logical_and((diff_t < diff_t_cut), training_labels[:,1] < logg_cut)
        return bad

    def get_stars(is_training, label_names):
        ' Constructs and returns a Stars object '
        if is_training:
            files = get_training_files()
            training_labels, apogee_labels = get_training_labels()
        else:
            files = get_test_files()
            training_labels = None
        spectra = get_spectra(files)
        cont_norm_spectra, continua = continuum_normalize(spectra)
        stars = Stars(files, cont_norm_spectra, [label_names, training_labels])
        to_discard = None
        if is_training: # because the condition depends on the labels...
            to_discard = discard_stars(training_labels, apogee_labels)
            stars.remove_stars(to_discard)
        return stars, to_discard
