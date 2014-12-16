' This is the file that the user will interact with. '
' The purpose of the file is to read in the raw data and create a Star object, which consists of continuum-normalized spectra (fluxes, flux errs) as well as training labels. '
' The only one of these functions that The Cannon will interact with is getStars '
' This assumes that you have two kinds of files: a list of continuum pixels (determined in whatever way you want), and a file with training labels '

from star import Star
import pyfits
import numpy as np
import os

npixels = 0
nstars = 0

def getSpectra(files):
    ' Reads file list and returns spectra array, shape (npixels, nstars, 3) '
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

def findGaps(spectra):

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

def continuumNormalize(spectra):
    ' For ASPCAP data, we fit a 2nd order Chebyshev polynomial to each segment of the continuum, and divide that segment by it '

    continua = np.zeros((nstars, npixels))
    normalized_spectra = np.ones((nstars, npixels, 3))
   
    # pixlist is a list of "true" continuum pixels, as determined in this case by the Cannon method
    pixlist = np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1)
    pixlist = list(pixlist)

    # Check for unphysical values
    for jj in range(0, nstars):
        isinf = np.isnan(spectra[jj,:,1]) |  np.isinf(spectra[jj,:,1]) | np.isnan(spectra[jj,:,2])
        negerr = (spectra[jj,:,2] <= 0)
        bad = isinf | negerr
        spectra[jj,:,1][bad] = 0.
        spectra[jj,:,2][bad] = np.Inf
  
    # Construct a variance array: [10,000] of length npixels
    var_array = 100**2*np.ones(npixels)
    var_array[pixlist] = 0.000
    ivar = 1. / ((spectra[:, :, 2] ** 2) + var_array)
    bad = np.isnan(ivar) | np.isinf(ivar)
    ivar[bad] = 0

    # We discard the edges of the fluxes, say 10 Angstroms, which is ~50 pixels
    ## The regions with flux were found to be: [321, 3242] [3647, 6047], [6411, 8305]
    ## With edge cuts: [371, 3192], [3697, 5997], [6461, 8255]
    ## Corresponding to: [15218, 15743] [15931, 16367] [16549, 16887] 

    split_spectrum = [spectra[:,371:3192,:], spectra[:,3697:5997,:], spectra[:,6461:8255,:]]
    split_ivar = [ivar[:,371:3192], ivar[:,3697:5997], ivar[:,6461:8255]]

    # Fit the Chebyshev polynomial
    for i in range(len(split_spectrum)):
        spectrum = split_spectrum[i]
        ivar = split_ivar[i]
        for jj in range(nstars):
            fit = np.polynomial.chebyshev.Chebyshev.fit(x=spectrum[jj,:,0], y=spectrum[jj,:,1], w=ivar[jj], deg=3)
            continuum = fit(spectrum[jj,:,0])
            normalized_fluxes = spectrum[jj,:,1]/fit(spectrum[0,:,0])
            normalized_flux_errs = spectrum[jj,:,2]/fit(spectrum[0,:,0])

    #continuum = fit

# These two functions are currently exactly the same because of the nature of the APOGEE set...this is for a sanity check where the test set == the training set
def getTrainingFiles():
    ' Return: filenames array of length (ntrainingstars) '
    readin = "starsin_SFD_Pleiades.txt"
    filenames = np.loadtxt(readin, usecols = (0,), dtype='string', unpack = 1)
    filenames1 = [] # for some reason if I try to replace the element, it gets rid of the '.fits' at the end...
    for i in range(0, len(filenames)): # incorporate file location info
        filename = '../Data/APOGEE_Data' + filenames[i][1:] 
        filenames1.append(filename)
    return filenames1

def getTestFiles():
    ' Return: filenames array of length (nteststars) ' 
    readin = "starsin_SFD_Pleiades.txt"
    filenames = np.loadtxt(readin, usecols = (0,), dtype='string', unpack = 1)
    filenames1 = [] # for some reason if I try to replace the element, it gets rid
    for i in range(0, len(filenames)):
        filename = '../Data/APOGEE_Data' + filenames[i][1:] 
        filenames1.append(filename)
    return filenames1

def getTrainingLabels():
    ' Return: 2D array of size (nlabels, ntrainingstars) '
    input1 = "starsin_SFD_Pleiades.txt"
    input2 = "ages_2.txt"
    T_est,g_est,feh_est,T_A, g_A, feh_A = np.loadtxt(input1, usecols = (4,6,8,3,5,7), unpack =1)
    age_est = np.loadtxt(input2, usecols = (0,), unpack =1)
    training_labels = np.array([T_est, g_est, feh_est, age_est])
    return training_labels

def getStars(isTraining):
    ' Constructs and returns a Stars object '
    if isTraining:
        files = getTrainingFiles()
        training_labels = getTrainingLabels()
    else:
        files = getTestFiles()
        training_labels = None
    nstars = len(files)
    spectra = getSpectra(files)
    cont_norm_spectra = continuumNormalize(spectra)
    stars = Stars(files, cont_norm_spectra, training_labels)
    return stars
