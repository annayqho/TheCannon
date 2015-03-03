# Experiment with continuum normalization.

import pyfits
import os
import numpy as np
import matplotlib.pyplot as plt

def get_pixmask(fluxes, flux_errs):
  bad_flux = (~np.isfinite(fluxes)) | (fluxes == 0)
  bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
  bad_pix = bad_err | bad_flux
  return bad_pix

def get_contmask(lambdas, fluxes, f_cut, sigma_cut):
  ### Use the lambdas and fluxes to find continuum pixels
    f_bar = np.median(fluxes, axis=0)
    sigma_f = np.var(fluxes, axis=0)
    pixmask = get_pixmask(f_bar, sigma_f)
    numpix = len(pixmask) - sum(pixmask) # 7212

    ### Aim for roughly 5% of 7212, which is about 360

    #f_bar[pixmask] = 0.
    #sigma_f[pixmask] = LARGE

    cont1 = np.abs(f_bar-1)/1 <= f_cut
    cont2 = sigma_f <= sigma_cut
    cont3 = sigma_f >= np.abs(1-f_bar)
    contmask1 = np.logical_and(cont1, cont2)
    contmask = np.logical_and(contmask1, cont3)

    ### Plot the continuum pixels and see how they look

    contpix = lambdas[contmask]
    ones = np.ones(len(contpix))
    plt.plot(lambdas, f_bar, alpha=0.5)
    plt.scatter(contpix, f_bar[contmask], color='r', s=2)
    plt.errorbar(contpix, f_bar[contmask], yerr=sigma_f[contmask], fmt=None)
    plt.xlim(min(contpix), max(contpix))
    plt.ylim(0.90, 1.10)
    #plt.show()

    return contmask

if __name__ == "__main__":
    spec_dir = "example_DR10/Data"
    files = [spec_dir + "/" + filename for filename in os.listdir(spec_dir)]
    files = list(sorted(files))
    nstars = len(files)
    LARGE = 1000000.

    # Make the arrays
    for jj, fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        flux = np.array(file_in[1].data)
        if jj == 0:
            npixels = len(flux)
            SNRs = np.zeros(nstars, dtype=float)
            fluxes = np.zeros((nstars, npixels), dtype=float)
            flux_errs = np.zeros(fluxes.shape, dtype=float) 
            ivars = np.zeros(fluxes.shape, dtype=float)
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl * (npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10 ** aval for aval in wl_full_log]
            lambdas = np.array(wl_full)
        flux_err = np.array((file_in[2].data))
        badpix = get_pixmask(flux, flux_err)
        flux = np.ma.array(flux, mask=badpix, fill_value=0.)
        flux_err = np.ma.array(flux_err, mask=badpix, fill_value=LARGE)
        SNRs[jj] = np.ma.median(flux/flux_err)
        ones = np.ma.array(np.ones(npixels), mask=badpix)
        flux = np.ma.filled(flux)
        flux_err = np.ma.filled(flux_err)
        ivar = ones / (flux_err**2)
        ivar = np.ma.filled(ivar, fill_value=0.)
        fluxes[jj,:] = flux
        flux_errs[jj,:] = flux_err
        ivars[jj,:] = ivar

    f_cuts = np.linspace(0.001, 0.008, 8)
    sigma_cuts = np.linspace(0.001, 0.008, 8)
    npix = np.zeros((8,8))
    
    for i in range(0, 8):
        f_cut = f_cuts[i]
        for j in range(0, 8):
            sigma_cut = sigma_cuts[j]
            contmask = get_contmask(lambdas, fluxes, f_cut, sigma_cut)
            npix[i,j] = sum(contmask)

    # Plot the results
    cax = plt.imshow(npix, interpolation="nearest")
    cbar = plt.colorbar(cax, label="Number of Continuum Pixels")
    plt.xlabel("fbar cut*1000")
    plt.ylabel("fsig cut*1000")
    plt.show()
    plt.savefig("flux_cut_experimentation.png")
