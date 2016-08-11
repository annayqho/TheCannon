""" Pull out the APOKASC/LAMOST overlap set,
and create model spectra """

import pyfits
import os
import numpy as np
import glob
from TheCannon import lamost
from TheCannon import dataset
from model_spectra import get_model_spec

DATA_DIR = "/Users/annaho/Data/Li_Giants"
a = pyfits.open(DATA_DIR + "/" + "apokasc_lamost_overlap.fits")
data = a[1].data
a.close()

wl = np.load("/Users/annaho/Data/LAMOST/wl.npz")['arr_0']
lamost_id = data['lamost_id_2']


def get_model():
    """ Cannon model params """
    coeffs = np.load("/Users/annaho/Data/LAMOST/coeffs.npz")['arr_0']
    scatters = np.load("/Users/annaho/Data/LAMOST/scatters.npz")['arr_0']
    chisqs = np.load("/Users/annaho/Data/LAMOST/chisqs.npz")['arr_0']
    pivots = np.load("/Users/annaho/Data/LAMOST/pivots.npz")['arr_0']
    return coeffs, scatters, chisqs, pivots


def get_labels():
    """ Labels to make Cannon model spectra """
    cannon_teff = data['cannon_teff_2']
    cannon_logg = data['cannon_logg_2']
    cannon_m_h = data['cannon_m_h']
    cannon_alpha_m = data['cannon_alpha_m']
    cannon_a_k = data['cannon_a_k']
    labels = np.vstack(
            (cannon_teff, cannon_logg, cannon_m_h, cannon_alpha_m, cannon_a_k))
    cannon_chisq = data['cannon_chisq']
    np.savez(DATA_DIR + "chisq.npz", labels)
    np.savez(DATA_DIR + "labels.npz", labels)
    snrg = data['cannon_snrg'] # snrg * 3
    np.savez("snr.npz", snrg)
    return labels.T


def get_normed_spectra():
    """ Spectra to compare with models """
    filenames = np.array(
            [DATA_DIR + "/Spectra" + "/" + val for val in lamost_id])
    grid, fluxes, ivars, npix, SNRs = lamost.load_spectra(
            lamost_id, input_grid=wl)
    ds = dataset.Dataset(
            wl, lamost_id, fluxes, ivars, [1], 
            lamost_id[0:2], fluxes[0:2], ivars[0:2])
    ds.continuum_normalize_gaussian_smoothing(L=50)
    np.savez(DATA_DIR + "/" + "norm_flux.npz", ds.tr_flux)
    np.savez(DATA_DIR + "/" + "norm_ivar.npz", ds.tr_ivar)
    return ds.tr_flux, ds.tr_ivar


def wget_files():
    """ Pull the files from the LAMOST archive """
    for f in lamost_id:
        short = (f.split('-')[2]).split('_')[0]
        filename = "%s/%s.gz" %(short,f)
        DIR = "/Users/annaho/Data/Li_Giants/Spectra_APOKASC"
        searchfor = "%s/%s.gz" %(DIR,f)
        if glob.glob(searchfor):
            print("done")
        else:
            print(searchfor)
            os.system(
                    "wget http://dr2.lamost.org/sas/fits/%s" %(filename))
            new_filename = filename.split("_")[0] + "_" + filename.split("_")[2]
            os.system(
                    "wget http://dr2.lamost.org/sas/fits/%s" %(new_filename))
            #spec-56094-kepler05B56094_2_sp10-118.fits.gz


if __name__=="__main__":
    wget_files()
    # labels = get_labels()
    # norm_flux, norm_ivar = get_normed_spectra()
    # coeffs, scatters, chisqs, pivots = get_model()
    # model_spec = get_model_spec(
    #         wl, labels, coeffs, scatters, chisqs, pivots)
    # nobj = len(model_spec)
    # for ii in np.arange(nobj):
    #     plt.plot(wl, norm_flux[ii]-model_spec, c='k')
    #     plt.savefig("resid_%s.png" %ii)
    #     plt.close()
