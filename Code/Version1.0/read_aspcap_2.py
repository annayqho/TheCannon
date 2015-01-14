"""
This is the ASPCAP implementation of the ReadData class, which establishes 
the training set and test set for input into The Cannon.

ReadASPCAP inherits from ReadData. The user (me, in this case) has tailored 
each method to suit the particularly survey and data file format. The only 
methods that the user "sees" are the ones that he needs to write. For example, 
set_star_set does not appear here even though it is in ReadData, because 
it should be survey-dependent.  
"""

from read_data_2 import ReadData
from stars import Stars
import pyfits
import numpy as np
import os

class ReadASPCAP(ReadData):

    def __init__(self):
        ReadData.__init__(self)

    def get_spectra(self, files):
        """
        Extracts spectra (wavelengths, fluxes, fluxerrs) from aspcap fits files

        Input: a list of data file names of length nstars
        Returns: a 3D float array of shape (nstars, npixels, 3)
        with spectra[:,:,0] = pixel wavelengths
        spectra[:,:,1] = flux values
        spectra[:,:,2] = flux err values
        """
        for jj,fits_file in enumerate(files):
            file_in = pyfits.open(fits_file)
            fluxes = np.array(file_in[1].data)
            if jj == 0: 
                nstars = len(files) 
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

    def find_gaps(self, spectra):
        """
        HWR, MKN, you can ignore this. 
        AH used this to check the wavelength regions found by Melissa.
        This would probably be part of the continuum normalization. 
        """
        # Find gaps in the spectrum, assume series of zeros with len(>=50 pix)
        # Are there stars for which the gap is not the same?  
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

    def continuum_normalize(self, spectra):
        """
        Continuum-normalizes the spectra.

        Fit a 2nd order Chebyshev polynomial to each segment 
        and divide each segment by its corresponding polynomial 

        Input: spectra array, 2D float shape nstars,npixels,3
        Returns: 3D continuum-normalized spectra (nstars, npixels,3)
                2D continuum array (nstars, npixels)
        """
        nstars = spectra.shape[0]
        npixels = spectra.shape[1]
        continua = np.zeros((nstars, npixels))
        normalized_spectra = np.ones((nstars, npixels, 3))
        # list of "true" continuum pix, det. here by the Cannon
        pixlist = list(np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1))
        # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels
        ## I found the regions with flux to be: 
        ## [321, 3242] [3647, 6047], [6411, 8305]
        ## With edge cuts: [371, 3192], [3697, 5997], [6461, 8255]
        ## Corresponding to: [15218, 15743] [15931, 16367] [16549, 16887] 
        ranges = [[371,3192], [3697,5997], [6461,8255]]
        LARGE = 200.
        for jj in range(nstars):
            # There must be a better way to do this, but right now I'm stumped
            # Mask unphysical pixels
            bad1 = np.invert(np.logical_and(np.isfinite(spectra[jj,:,1]),  
                np.isfinite(spectra[jj,:,2])))
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
                fit = np.polynomial.chebyshev.Chebyshev.fit(x=spectrum[:,0], 
                        y=spectrum[:,1], w=ivar1, deg=3)
                continua[jj,start:stop] = fit(spectrum[:,0])
                normalized_fluxes = spectrum[:,1]/fit(spectra[0,start:stop,0])
                bad = np.invert(np.isfinite(normalized_fluxes))
                normalized_fluxes[bad] = 1.
                normalized_flux_errs = spectrum[:,2]/fit(spectra[0,start:stop,0])
                bad = np.logical_or(np.invert(np.isfinite(normalized_flux_errs)),
                        normalized_flux_errs <= 0)
                normalized_flux_errs[bad] = LARGE
                normalized_spectra[jj,:,0] = spectra[jj,:,0]
                normalized_spectra[jj,start:stop,1] = normalized_fluxes 
                normalized_spectra[jj,start:stop,2] = normalized_flux_errs
            # One last check for unphysical pixels
            bad = spectra[jj,:,2] > LARGE
            normalized_spectra[jj,np.logical_or(bad, bad1),1] = 1.
            normalized_spectra[jj,np.logical_or(bad, bad1),2] = LARGE
        return normalized_spectra, continua

    # Note: the following two functions are identical only because
    # for the GC check, the test set == the training set
    
    def get_training_files(self):
        """
        Establishes which files correspond to the training set data.
        Returns: an array of filenames of length ntrainingstars
        """
        readin = "starsin_SFD_Pleiades.txt"
        filenames = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
        filenames1 = [] # for some reason if I try to replace the element, 
                        # it gets rid of the '.fits' at the end...very annoying
        for i in range(0, len(filenames)): # incorporate file location info
            filename = '../Data/APOGEE_Data' + filenames[i][1:] 
            filenames1.append(filename)
        return np.array(filenames1)

    def get_test_files(self):
        """
        Establishes which files correspond to the test set data.
        Returns: an array of string filenames of length nteststars
        """
        readin = "starsin_SFD_Pleiades.txt"
        filenames = np.loadtxt(readin, usecols = (0,), dtype='string', unpack = 1)
        filenames1 = [] # for some reason if I try to replace the element, 
                        # it gets rid of the '.fits' at the end...very annoying
        for i in range(0, len(filenames)): # incorporate file location info
            filename = '../Data/APOGEE_Data' + filenames[i][1:] 
            filenames1.append(filename)
        return np.array(filenames1)

    def get_training_labels(self):
        """
        Extracts training labels from file

        Input: name(string) of the data file containing the labels
        Returns: a 2D np.array (size=numtrainingstars, nlabels)
        consisting of all of the training labels
        """
        input1 = "starsin_SFD_Pleiades.txt"
        input2 = "ages_2.txt"
        #input2 = "logAges.txt"
        T_est,g_est,feh_est,T_A, g_A, feh_A = np.loadtxt(input1, 
                usecols = (4,6,8,3,5,7), unpack =1)
        age_est = np.loadtxt(input2, usecols = (0,), unpack =1)
        training_labels = np.array([T_est, g_est, feh_est, age_est])
        # This is a cheat...not sure how to deal with this generally.
        # this should probably be dealt with somehow by discard_stars
        apogee_labels = np.array([T_A, g_A, feh_A])
        return training_labels.T, apogee_labels.T

    def set_stars_to_discard(self, training_labels, apogee_labels):
        """
        Create a mask indicating which stars to throw out.
        aspcap Criteria: logg cut, diff_t cut

        Returns: a boolean array of len(nstars) where True means 'discard'
        """
        diff_t = np.abs(apogee_labels[:,0]-training_labels[:,0]) # temp difference
        # How were these determined..? What should these be? 
        logg_cut = 100.
        diff_t_cut = 600.
        bad = np.logical_and((diff_t < diff_t_cut), 
                training_labels[:,1] < logg_cut)
        return bad
    
    # This method should not be here, but I'm not sure how to make general
    # the discard_stars requirements we have imposed here. 
    def set_star_set(self, is_training, label_names):
        """
        Constructs and returns a Stars object, which consists of a 
        set of spectra and (if training set) training labels, using 
        the methods defined above by the user.

        Input: is_training boolean (True or False), True if training set
        Input: label_names, a list of strings (ex. ['FeH', 'logg'])
        Returns: a Stars object
        """
        if is_training:
            files = self.get_training_files()
            training_labels, apogee_labels = self.get_training_labels()
        else:
            files = self.get_test_files()
            training_labels = None
        spectra = self.get_spectra(files)
        cont_norm_spectra, continua = self.continuum_normalize(spectra)
        stars = Stars(files, cont_norm_spectra, [label_names, training_labels])
        to_discard = None
        if is_training: # because the condition depends on the labels...
            to_discard = self.set_stars_to_discard(training_labels, apogee_labels)
            stars.remove_stars(to_discard)
        return stars, to_discard
