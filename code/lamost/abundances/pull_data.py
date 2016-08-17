""" Generate the files to run the abundances paper """

import pyfits
import numpy as np
from TheCannon import dataset
import sys
sys.path.append("/Users/annaho/Dropbox/Research/TheCannon/code/lamost")
from get_colors import get_colors


def load_all_ref_label():
    DATA_DIR = "/Users/annaho/Data/LAMOST/Abundances"
    a = pyfits.open(
            DATA_DIR + "/casey_lamost_paper_one_cross_match_with_colors.fits")
    tbl = a[1].data
    a.close()
    ref_id = tbl['lamost_id']
    ref_id = np.array(ref_id)
    ref_id = np.array([val.strip() for val in ref_id])
    snrg = tbl['snrg']
    labels = ['TEFF', 'LOGG', 'AK_WISE',
            'AL_H', 'CA_H', 'C_H', 'FE_H', 'MG_H', 'MN_H',
            'NI_H', 'N_H', 'O_H', 'SI_H', 'TI_H']
    nlabel = len(labels)
    nobj = len(ref_id)
    ref_label = np.zeros((nobj, nlabel))
    for ii,label in enumerate(labels):
        ref_label[:,ii] = tbl[label]
    np.savez("ref_id.npz", ref_id)
    np.savez("ref_label.npz", ref_label)
    return ref_id


def load_all_ref_spectra(ref_id):
    DATA_DIR = "/Users/annaho/Data/LAMOST/Label_Transfer"
    wl = np.load(DATA_DIR + "/../Abundances/wl_cols.npz")['arr_0']
    all_ref_ivar = np.load("%s/tr_ivar.npz" %DATA_DIR)['arr_0']
    all_ref_flux = np.load("%s/tr_flux.npz" %DATA_DIR)['arr_0']
    all_id = np.load("%s/tr_id.npz" %DATA_DIR)['arr_0']
    all_id = np.array([val.decode('utf-8') for val in all_id])
    inds = np.array([np.where(all_id==val)[0][0] for val in ref_id])
    ref_flux = all_ref_flux[inds]
    ref_ivar = all_ref_ivar[inds]

    mask = np.load("%s/../Abundances/mask.npz" %DATA_DIR)['arr_0']
    ref_ivar_masked = apply_mask(wl[0:3626], ref_ivar, mask)
    ref_id_col, ref_flux_col, ref_ivar_col = find_colors(
            ref_id, ref_flux, ref_ivar_masked)
    np.savez("ref_id_col.npz", ref_id_col)
    np.savez("ref_flux.npz", ref_flux_col)
    np.savez("ref_ivar.npz", ref_ivar_col)
    ds = dataset.Dataset(
            wl[0:3626], ref_id_col, ref_flux_col[:,3626], ref_ivar_col[:,3626], 
            [], [], [], [])
    np.savez("ref_snr.npz", ds.tr_SNR)


def apply_mask(wl, ref_ivar, mask):
    # Mask out wl
    # Mask out tellurics, DIBs, the Na double, the end of hte spectrum
    print("Applying mask")
    label_names = ['T_{eff}', '\log g', '[M/H]', '[C/M]', '[N/M]', 
            '[\\alpha/M]', 'Ak']
    ref_ivar[:,mask] = 0.0
    end = wl > 8750
    ref_ivar[:,end] = 0.0
    return ref_ivar


def add_to_wl():
    # Add wavelengths to wl
    for col in np.arange(ncol):
        delt = ((wl[1:]-wl[:-1] )/ (wl[1:] + wl[:-1]))[0]
        new_wl = (wl[-1]*delt + wl[-1]) / (1-delt)
        wl = np.append(wl, new_wl)
    np.savez("wl_cols.npz", wl)


def find_colors(ref_id, ref_flux, ref_ivar):
    # Find colors
    DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"
    print("Finding colors")
    a = pyfits.open(DATA_DIR + "/lamost_catalog_colors.fits")
    data = a[1].data
    a.close()
    all_ids = data['LAMOST_ID_1']
    all_ids = np.array([val.strip() for val in all_ids])
    ref_id_col = np.intersect1d(all_ids, ref_id)
    inds = np.array([np.where(all_ids==val)[0][0] for val in ref_id_col])
    all_id, all_col, all_col_ivar = get_colors(
            DATA_DIR + "/lamost_catalog_colors.fits")
    col = all_col[:,inds]
    col_ivar = all_col_ivar[:,inds]
    bad_ivar = np.logical_or(np.isnan(col_ivar), col_ivar==np.inf)
    col_ivar[bad_ivar] = 0.0
    bad_flux = np.logical_or(np.isnan(col), col==np.inf)
    col[bad_flux] = 1.0
    col_ivar[bad_flux] = 0.0
    # add them to the wl, flux and ivar arrays
    inds = np.array([np.where(ref_id==val)[0][0] for val in ref_id_col])
    ref_flux_col = np.hstack((ref_flux[inds], col.T))
    ref_ivar_col = np.hstack((ref_ivar[inds], col_ivar.T))
    return ref_id_col, ref_flux_col, ref_ivar_col


if __name__=="__main__":
    ref_id = load_all_ref_label()
    load_all_ref_spectra(ref_id)
