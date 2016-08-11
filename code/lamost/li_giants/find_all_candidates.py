""" Pull out the APOKASC/LAMOST overlap set,
and create model spectra """

import pyfits
import os
import numpy as np
import glob
from TheCannon import lamost
from TheCannon import dataset
from model_spectra import get_model_spec
from model_spectra import spectral_model

SPEC_DIR = "/Users/annaho/Data/Li_Giants/All_Vals"
LAB_DIR = "/Users/annaho/Data/LAMOST"


def get_model():
    """ Cannon model params """
    coeffs = np.load("%s/coeffs.npz" %LAB_DIR)['arr_0']
    scatters = np.load("%s/scatters.npz" %LAB_DIR)['arr_0']
    chisqs = np.load("%s/chisqs.npz" %LAB_DIR)['arr_0']
    pivots = np.load("%s/pivots.npz" %LAB_DIR)['arr_0']
    return coeffs, scatters, chisqs, pivots


def get_labels(ids_find):
    """ Labels to make Cannon model spectra """
    a = pyfits.open("%s/lamost_catalog_full.fits" %LAB_DIR)
    data = a[1].data
    a.close()
    id_all = data['lamost_id']
    id_all = np.array(id_all)
    id_all = np.array([val.strip() for val in id_all])
    snr_all = data['cannon_snrg']
    chisq_all = data['cannon_chisq']
    teff = data['cannon_teff']
    logg = data['cannon_logg']
    feh = data['cannon_m_h']
    afe = data['cannon_alpha_m']
    ak = data['cannon_a_k']
    labels = np.vstack((teff,logg,feh,afe,ak))
    choose = np.in1d(id_all, ids_find)
    id_choose = id_all[choose]
    label_choose = labels[:,choose]
    snr_choose = snr_all[choose]
    chisq_choose = chisq_all[choose]
    inds = np.array([np.where(id_choose==val)[0][0] for val in ids_find])
    print(id_choose[inds][100])
    print(ids_find[100])
    return label_choose[:,inds], snr_choose[inds], chisq_choose[inds]


def get_normed_spectra():
    """ Spectra to compare with models """
    wl = np.load("%s/wl.npz" %LAB_DIR)['arr_0']
    filenames = np.array(
            [SPEC_DIR + "/Spectra" + "/" + val for val in lamost_id])
    grid, fluxes, ivars, npix, SNRs = lamost.load_spectra(
            lamost_id, input_grid=wl)
    ds = dataset.Dataset(
            wl, lamost_id, fluxes, ivars, [1], 
            lamost_id[0:2], fluxes[0:2], ivars[0:2])
    ds.continuum_normalize_gaussian_smoothing(L=50)
    np.savez(SPEC_DIR + "/" + "norm_flux.npz", ds.tr_flux)
    np.savez(SPEC_DIR + "/" + "norm_ivar.npz", ds.tr_ivar)
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
            #print(searchfor)
            os.system(
                    "wget http://dr2.lamost.org/sas/fits/%s" %(filename))
            new_filename = filename.split("_")[0] + "_" + filename.split("_")[2]
            os.system(
                    "wget http://dr2.lamost.org/sas/fits/%s" %(new_filename))
            #spec-56094-kepler05B56094_2_sp10-118.fits.gz


if __name__=="__main__":
    ids = np.load(SPEC_DIR + "/ids.npz")['arr_0']
    labels,snr,chisq = get_labels(ids)
    wl = np.load("%s/wl.npz" %LAB_DIR)['arr_0']
    flux = np.load(SPEC_DIR + "/norm_flux.npz")['arr_0']
    ivar = np.load(SPEC_DIR + "/norm_ivar.npz")['arr_0']

    coeffs, scatters, chisqs, pivots = get_model()
    model_spec = get_model_spec(
            wl, labels.T, coeffs, scatters, chisqs, pivots)
    good_chisq = np.logical_and(chisq > 500, chisq < 3000)
    good = np.logical_and(snr > 200, good_chisq)
    # print(sum(good))
    nobj = len(model_spec[good])
    # for ii in np.arange(nobj):
    #     print(ii/nobj)
    #     spectral_model(
    #             ii, wl, flux[good], ivar[good], model_spec[good], 
    #             coeffs, scatters, chisqs, pivots)
    # # look for Li residual
    # good_resid = resid[good]
    # li_region = np.where(np.logical_and(wl > 6700, wl < 6720))[0]
    # li_resid = resid[:,li_region][good]
    # max_resid = np.max(li_resid, axis=1)
    # order = np.argsort(max_resid)
    # resid_order = li_resid[order]
    # 

    #nobj = len(model_spec)
    # for ii in np.arange(nobj):
    #     plt.plot(wl, norm_flux[ii]-model_spec, c='k')
    #     plt.savefig("resid_%s.png" %ii)
    #     plt.close()
