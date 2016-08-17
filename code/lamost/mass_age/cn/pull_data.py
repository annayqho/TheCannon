""" Generate the files to run the mass & age + photometry paper """

import pyfits
import numpy as np
from TheCannon import dataset
from get_colors import get_colors

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"

def load_all_ref():
    wl = np.load("%s/no_colors/wl.npz" %DATA_DIR)['arr_0']
    print(wl.shape)
    ref_id = np.load("%s/no_colors/ref_id_col.npz" %DATA_DIR)['arr_0']
    ref_id = np.array([val.strip() for val in ref_id])
    print(ref_id.shape)
    ref_flux = np.load("%s/no_colors/ref_flux.npz" %DATA_DIR)['arr_0']
    print(ref_flux.shape)
    ref_ivar = np.load("%s/no_colors/ref_ivar.npz" %DATA_DIR)['arr_0']
    ref_ivar_masked = apply_mask(wl, ref_ivar)
    print(ref_ivar.shape)
    ref_label = np.load("%s/no_colors/ref_label.npz" %DATA_DIR)['arr_0']
    print(ref_label.shape)
    ref_id_col, ref_flux_col, ref_ivar_col = find_colors(
            ref_id, ref_flux, ref_ivar_masked)
    inds = np.array([np.where(ref_id==val)[0][0] for val in ref_id_col])
    #np.savez("ref_id_col.npz", ref_id_col)
    #np.savez("ref_flux_col.npz", ref_flux_col)
    #np.savez("ref_ivar_col.npz", ref_ivar_col)
    #np.savez("ref_label.npz", ref_label[inds])
    cn_band = np.logical_and(wl > 4100, wl < 4250)
    inds = np.where(cn_band)[0]
    start = inds[0]
    end = inds[-1]
    #ds = dataset.Dataset(
    #        wl[0:3626], ref_id_col, ref_flux_col[:,3626], ref_ivar_col[:,3626], 
    #        ref_label, [], [], [])
    ds = dataset.Dataset(
            wl[start:end], ref_id_col, ref_flux_col[:,start:end], 
            ref_ivar_col[:,start:end], ref_label, [], [], [])
    np.savez("cn_ref_snr.npz", ds.tr_SNR)


def apply_mask(wl, ref_ivar, mask):
    # Mask out wl
    # Mask out tellurics, DIBs, the Na double, the end of hte spectrum
    print("Applying mask")
    #mask = np.load("%s/mask.npz" %DATA_DIR)['arr_0']
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
    print("Finding colors")
    a = pyfits.open(DATA_DIR + "/lamost_catalog_colors.fits")
    data = a[1].data
    a.close()
    all_ids = data['LAMOST_ID_1']
    all_ids = np.array([val.strip() for val in all_ids])
    ref_id_col = np.intersect1d(all_ids, ref_id)
    inds = np.array([np.where(all_ids==val)[0][0] for val in ref_id_col])
    all_col, all_col_ivar = get_colors(
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
    load_all_ref()
