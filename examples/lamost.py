""" Code for LAMOST data munging """

from __future__ import (absolute_import, division, print_function,)
from apogee import get_starmask
import numpy as np
import scipy.optimize as opt
from scipy import interpolate 
import os
import sys
import matplotlib.pyplot as plt
import glob

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

def get_pixmask(file_in, wl, middle, flux, ivar):
    """ Return a mask array of bad pixels for one object's spectrum

    Bad pixels are defined as follows: fluxes or ivars are not finite, or 
    ivars are negative

    Major sky lines. 4046, 4358, 5460, 5577, 6300, 6363, 6863

    Where the red and blue wings join together: 5800-6000

    Read bad pix mask: file_in[0].data[3] is the andmask 

    Parameters
    ----------
    fluxes: ndarray
        flux array

    flux_errs: ndarray
        measurement uncertainties on fluxes

    Returns
    -------
    mask: ndarray, dtype=bool
        array giving bad pixels as True values
    """
    npix = len(wl)
    
    bad_flux = (~np.isfinite(flux)) # count: 0
    bad_err = (~np.isfinite(ivar)) | (ivar <= 0)
    # ivar == 0 for approximately 3-5% of pixels
    bad_pix_a = bad_err | bad_flux
    
    # LAMOST people: wings join together, 5800-6000 Angstroms
    wings = np.logical_and(wl > 5800, wl < 6000)
    # this is another 3-4% of the spectrum
    andmask = (file_in[0].data[3] >0)
    # ^ problematic...this is over a third of the spectrum!
    bad_pix_b = wings | andmask
    # bad_pix_b = wings

    spread = 3 # due to redshift
    skylines = np.array([4046, 4358, 5460, 5577, 6300, 6363, 6863])
    bad_pix_c = np.zeros(npix, dtype=bool)
    for skyline in skylines:
        badmin = skyline-spread
        badmax = skyline+spread
        bad_pix_temp = np.logical_and(wl > badmin, wl < badmax)
        bad_pix_c[bad_pix_temp] = True
    # 34 pixels

    bad_pix_ab = bad_pix_a | bad_pix_b
    bad_pix = bad_pix_ab | bad_pix_c

    return bad_pix


def load_spectra(data_dir, filenames):
    """
    Extracts spectra (wavelengths, fluxes, fluxerrs) from apogee fits files

    Returns
    -------
    IDs: list of length nstars
        stellar IDs
    
    wl: numpy ndarray of length npixels
        rest-frame wavelength vector

    fluxes: numpy ndarray of shape (nstars, npixels)
        training set or test set pixel intensities

    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes
        
    SNRs: numpy ndarray of length nstars
    """
    print("Loading spectra from directory %s" %data_dir)
    files = list(sorted(filenames))
    files = np.array(files)
    nstars = len(files)

    npix = np.zeros(nstars)
    for jj, fits_file in enumerate(files):
        file_in = pyfits.open("%s/%s" %(data_dir, fits_file))
        grid_all = np.array(file_in[0].data[2])
        if jj == 0:
            # all stars do NOT start out on the same wavelength grid
            middle = np.logical_and(grid_all > 3905, grid_all < 9000)
            grid = grid_all[middle]
            npixels = len(grid) 
            SNRs = np.zeros(nstars, dtype=float)   
            fluxes = np.zeros((nstars, npixels), dtype=float)
            ivars = np.zeros(fluxes.shape, dtype=float)
            #badpixs = np.zeros((nstars, len(grid_all)), dtype=bool)
        flux = np.array(file_in[0].data[0])
        npix[jj] = len(flux)
        ivar = np.array((file_in[0].data[1]))
        # identify bad pixels PRIOR to shifting, so that the sky lines
        # don't move around
        badpix = get_pixmask(file_in, grid_all, middle, flux, ivar)
        flux = np.ma.array(flux, mask=badpix)
        ivar = np.ma.array(ivar, mask=badpix)
        SNRs[jj] = np.ma.median(flux*ivar**0.5)
        ivar = np.ma.filled(ivar, fill_value=0.)
        # correct for radial velocity of star
        redshift = file_in[0].header['Z']
        wlshift = redshift*grid_all
        wl = grid_all - wlshift
        # resample onto a common grid
        flux_rs = (interpolate.interp1d(wl, flux))(grid)
        ivar_rs = (interpolate.interp1d(wl, ivar))(grid)
        fluxes[jj,:] = flux_rs
        ivars[jj,:] = ivar_rs

    print("Spectra loaded")
    return files, grid, fluxes, ivars


def load_labels(label_file, tr_files):
    """ Extracts training labels from file.

    Assumes that first row is # then label names, first col is # then 
    filenames, remaining values are floats and user wants all the labels.
    """
    print("Loading reference labels from file %s" %label_file)
    ids = np.loadtxt(
        label_file, usecols=(0,), delimiter=',', dtype=str)
    all_tr_label_val = np.loadtxt(
        label_file, usecols=(1,2,3,4), delimiter=',', dtype=str)
    tr_labels = np.zeros((len(tr_files), all_tr_label_val.shape[1]))
    for jj,tr_id in enumerate(tr_files):
        tr_labels[jj,:] = all_tr_label_val[ids==tr_id,:]
    tr_labels = tr_labels[np.argsort(tr_files)]
    return tr_labels


def is_badstar(star_id):
    ids = np.loadtxt(
        "apogee_dr12_labels.csv", usecols=(0,), delimiter=',', dtype=str)
    bad = np.loadtxt(
        "apogee_dr12_labels.csv", usecols=(6,), delimiter=',', dtype=str)
    return bad[ids==star_id]


def make_apogee_label_file():
    """ only for the 11,057 overlap objects """

    lamost_key = np.loadtxt('lamost_sorted_by_ra.txt',dtype=str)
    apogee_key = np.loadtxt('apogee_sorted_by_ra.txt', dtype=str)
    apogee_key_short = np.array([(item.split('v603-')[-1]).split('.fits')[0] 
                                for item in apogee_key])
    nstars = len(lamost_key)

    hdulist = pyfits.open("allStar-v603.fits")
    datain = hdulist[1].data
    apstarid= datain['APSTAR_ID']
    aspcapflag = datain['ASPCAPFLAG']
    paramflag =datain['PARAMFLAG']
    apogee_id = np.array([element.split('.')[-1] for element in apstarid])
    t = np.array(datain['TEFF'], dtype=float)
    g = np.array(datain['LOGG'], dtype=float)
    # according to Holzman et al 2015, the most reliable values
    f = np.array(datain['PARAM_M_H'], dtype=float)
    a = np.array(datain['PARAM_ALPHA_M'], dtype=float)
    vscat = np.array(datain['VSCATTER'])
    SNR = np.array(datain['SNR'])
    labels = np.vstack((t, g, f, a))

    # 1 if object would be an unsuitable training object
    mask = get_starmask(apogee_id, labels, aspcapflag, paramflag)

    # we only want the objects that are in apogee_key
    inds = np.array([np.where(apogee_id==ID)[0][0] for ID in apogee_key_short]) 
    teff = t[inds]
    logg = g[inds]
    feh = f[inds]
    alpha = a[inds]
    snr = SNR[inds]
    vscatter = vscat[inds]
    starflags = mask[inds]

    outputf = open("apogee_dr12_labels.csv", "w")
    header = "#lamost_id,apogee_id,teff,logg,feh,alpha,snr,vscatter,starflag\n"
    outputf.write(header)
    for i in range(nstars):
        line = lamost_key[i]+','+apogee_key[i]+','+\
               str(teff[i])+','+str(logg[i])+','+str(feh[i])+','+\
               str(alpha[i])+','+str(snr[i])+','+str(vscatter[i])+','+\
               str(starflags[i])+'\n'
        outputf.write(line)
    outputf.close()


def make_tr_file_list(frac_cut=0.94, snr_cut=100):
    """ make a list of training objects, given cuts

    Parameters
    ----------
    frac_cut: float
        the fraction of pix in the spectrum that must be good

    snr_cut: float
        the snr that the spec

    Returns
    -------
    tr_files: np array
        list of file names of training objects
    """
    allfiles = np.loadtxt(
            "apogee_dr12_labels.csv", delimiter=',', usecols=(0,), dtype=str)
    starflags = np.loadtxt(
            "apogee_dr12_labels.csv", delimiter=',', usecols=(8,), dtype=str)
    nstars = len(allfiles)
    dir_dat = "example_LAMOST/Data_All"
    ID, wl, flux, ivar = load_spectra(dir_dat, allfiles)
    npix = np.float(flux.shape[1])
    ngoodpix = np.array([np.count_nonzero(ivar[jj,:]) for jj in range(nstars)])
    good_frac = ngoodpix/npix
    SNR_raw = flux * ivar**0.5
    bad = SNR_raw == 0
    SNR_raw = np.ma.array(SNR_raw, mask=bad)
    SNR = np.ma.median(SNR_raw, axis=1).filled()
    good_cut = np.logical_and(good_frac > frac_cut, SNR>snr_cut)
    good = np.logical_and(good_cut, starflags=="False")
    tr_files = ID[good] #945 spectra 
    outputf = open("tr_files.txt", "w")
    for tr_file in tr_files:
        outputf.write(tr_file + '\n')
    outputf.close()
    return tr_files


if __name__ == '__main__':
    make_apogee_label_file()
