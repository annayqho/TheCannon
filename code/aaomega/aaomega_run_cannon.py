""" Apply The Cannon to the AAOmega Spectra! """

import numpy as np
import matplotlib.pyplot as plt
import sys
from TheCannon import dataset
from TheCannon import model

DATA_DIR = '/Users/annaho/Data/AAOmega/Run_13_July'
SMALL = 1.0 / 1000000000.0


def test_step_iteration(ds, md, starting_guess):
    errs, chisq = md.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def choose_reference_set():
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    all_id = np.load("%s/ref_id_all.npz" %DATA_DIR)['arr_0']
    all_flux = np.load("%s/ref_flux_all.npz" %DATA_DIR)['arr_0']
    all_scat = np.load("%s/ref_spec_scat_all.npz" %DATA_DIR)['arr_0']
    all_label = np.load("%s/ref_label_all.npz" %DATA_DIR)['arr_0']
    all_ivar = np.load("%s/ref_ivar_corr.npz" %DATA_DIR)['arr_0']

    # choose reference objects
    good_teff = np.logical_and(
            all_label[:,0] > 4000, all_label[:,0] < 6000)
    good_feh = np.logical_and(
            all_label[:,2] > -2, all_label[:,2] < 0.3)
    good_logg = np.logical_and(
            all_label[:,1] > 1, all_label[:,1] < 3)
    good_vrot = all_label[:,4] < 20.0
    good_scat = all_scat < 0.1

    good1 = np.logical_and(good_teff, good_feh)
    good2 = np.logical_and(good_logg, good_vrot)
    good12 = np.logical_and(good1, good2)
    good = np.logical_and(good12, good_scat)

    ref_id = all_id[good]
    print("%s objects chosen for reference set" %len(ref_id))
    ref_flux = all_flux[good]
    ref_ivar = all_ivar[good]
    ref_label = all_label[good]

    np.savez("%s/ref_id.npz" %DATA_DIR, ref_id)
    np.savez("%s/ref_flux.npz" %DATA_DIR, ref_flux)
    np.savez("%s/ref_ivar.npz" %DATA_DIR, ref_ivar)
    np.savez("%s/ref_label.npz" %DATA_DIR, ref_label)


def update_cont():
    contpix = np.load("wl_contpix_old.npz")['arr_0']
    # this array is a bit too long, clip it off
    contpix_new = contpix[np.logical_and(contpix>8420, contpix<8700)]

    inds = np.zeros(contpix_new.shape, dtype=int)
    for i,val in enumerate(contpix_new):
        # find the nearest pixel
        inds[i] = int(np.argmin(np.abs(wl-val)))
        
    contmask = np.zeros(len(wl), dtype=bool)
    contmask[inds] = 1
    np.savez("wl_contmask.npz", contmask)
    print("SAVED")



def normalize_ref_set():
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    ref_id = np.load("%s/ref_id.npz" %DATA_DIR)['arr_0']
    ref_flux = np.load("%s/ref_flux.npz" %DATA_DIR)['arr_0']
    ref_ivar = np.load("%s/ref_ivar.npz" %DATA_DIR)['arr_0']
    ref_label = np.load("%s/ref_label.npz" %DATA_DIR)['arr_0']

    ds = dataset.Dataset(
            wl, ref_id, ref_flux, ref_ivar, ref_label, 
            ref_id, ref_flux, ref_ivar)
    contmask = np.load("%s/wl_contmask.npz" %DATA_DIR)['arr_0']
    ds.set_continuum(contmask)

    cont = ds.fit_continuum(3, "sinusoid")
    np.savez("%s/ref_cont.npz" %DATA_DIR, cont)
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
            ds.continuum_normalize(cont)
    bad = np.logical_or(ref_flux <= 0, ref_flux > 1.1)
    norm_tr_ivar[bad] = 0.0
    np.savez("%s/ref_flux_norm.npz" %DATA_DIR, norm_tr_flux)
    np.savez("%s/ref_ivar_norm.npz" %DATA_DIR, norm_tr_ivar)


def normalize_test_set():
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    test_id = np.load("%s/test_id.npz" %DATA_DIR)['arr_0']
    test_flux = np.load("%s/test_flux.npz" %DATA_DIR)['arr_0']
    test_ivar = np.load("%s/test_ivar_corr.npz" %DATA_DIR)['arr_0']
    test_scat = np.load("%s/test_spec_scat.npz" %DATA_DIR)['arr_0']

    contmask = np.load("%s/wl_contmask.npz" %DATA_DIR)['arr_0']

    ds = dataset.Dataset(
            wl, test_id[0:2], test_flux[0:2], test_ivar[0:2], wl, 
            test_id, test_flux, test_ivar)
    ds.set_continuum(contmask)

    # For the sake of the normalization, no pixel with flux >= 3 sigma
    # should be continuum. 

    for ii,spec in enumerate(ds.test_flux): 
        err = test_scat[ii]
        bad = np.logical_and(
                ds.contmask == True, np.abs(1-spec) >= 3*err)
        ds.test_ivar[ii][bad] = SMALL

    cont = ds.fit_continuum(3, "sinusoid")
    np.savez("%s/test_cont.npz" %DATA_DIR, cont)
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
            ds.continuum_normalize(cont)
    bad = np.logical_or(test_flux <= 0, test_flux > 1.1)
    norm_test_ivar[bad] = 0.0
    np.savez("%s/test_flux_norm.npz" %DATA_DIR, norm_test_flux)
    np.savez("%s/test_ivar_norm.npz" %DATA_DIR, norm_test_ivar)


def choose_training_set():
    ref_id = np.load("%s/ref_id.npz" %DATA_DIR)['arr_0']
    ref_flux = np.load("%s/ref_flux_norm.npz" %DATA_DIR)['arr_0']
    ref_ivar = np.load("%s/ref_ivar_norm.npz" %DATA_DIR)['arr_0']
    ref_label = np.load("%s/ref_label.npz" %DATA_DIR)['arr_0']
    
    # randomly pick 80% of the objects to be the training set
    nobj = len(ref_id)
    assignments = np.random.randint(10, size=nobj)
    # if you're < 8, you're training
    choose = assignments < 8
    tr_id = ref_id[choose]
    tr_flux = ref_flux[choose]
    tr_ivar = ref_ivar[choose]
    tr_label = ref_label[choose]
    np.savez("%s/tr_id.npz" %DATA_DIR, tr_id)
    np.savez("%s/tr_flux_norm.npz" %DATA_DIR, tr_flux)
    np.savez("%s/tr_ivar_norm.npz" %DATA_DIR, tr_ivar)
    np.savez("%s/tr_label.npz" %DATA_DIR, tr_label)

    val_id = ref_id[~choose]
    val_flux = ref_flux[~choose]
    val_ivar = ref_ivar[~choose]
    val_label = ref_label[~choose]
    np.savez("%s/val_id.npz" %DATA_DIR, val_id)
    np.savez("%s/val_flux_norm.npz" %DATA_DIR, val_flux)
    np.savez("%s/val_ivar_norm.npz" %DATA_DIR, val_ivar)
    np.savez("%s/val_label.npz" %DATA_DIR, val_label)


def train():
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    tr_id = np.load("%s/tr_id.npz" %DATA_DIR)['arr_0']
    tr_flux = np.load("%s/tr_flux_norm.npz" %DATA_DIR)['arr_0']
    tr_ivar = np.load("%s/tr_ivar_norm.npz" %DATA_DIR)['arr_0']
    tr_label = np.load("%s/tr_label.npz" %DATA_DIR)['arr_0']
    val_id = np.load("%s/val_id.npz" %DATA_DIR)['arr_0']
    val_flux = np.load("%s/val_flux_norm.npz" %DATA_DIR)['arr_0']
    val_ivar = np.load("%s/val_ivar_norm.npz" %DATA_DIR)['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label[:,0:4], 
            val_id, val_flux, val_ivar)
    ds.set_label_names(["Teff", "logg", "FeH", 'aFe'])

    np.savez("%s/tr_SNR.npz" %DATA_DIR, ds.tr_SNR)

    fig = ds.diagnostics_SNR()
    plt.savefig("%s/SNR_dist.png" %DATA_DIR)
    plt.close()

    fig = ds.diagnostics_ref_labels()
    plt.savefig("%s/ref_label_triangle.png" %DATA_DIR)
    plt.close()

    md = model.CannonModel(2)
    md.fit(ds)

    fig = md.diagnostics_leading_coeffs(ds)
    plt.savefig("%s/leading_coeffs.png" %DATA_DIR)
    plt.close()

    np.savez("%s/coeffs.npz" %DATA_DIR, md.coeffs)
    np.savez("%s/scatters.npz" %DATA_DIR, md.scatters)
    np.savez("%s/chisqs.npz" %DATA_DIR, md.chisqs)
    np.savez("%s/pivots.npz" %DATA_DIR, md.pivots)


def validate():
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    tr_id = np.load("%s/tr_id.npz" %DATA_DIR)['arr_0']
    tr_flux = np.load("%s/tr_flux_norm.npz" %DATA_DIR)['arr_0']
    tr_ivar = np.load("%s/tr_ivar_norm.npz" %DATA_DIR)['arr_0']
    val_id = np.load("%s/val_id.npz" %DATA_DIR)['arr_0']
    val_flux = np.load("%s/val_flux_norm.npz" %DATA_DIR)['arr_0']
    val_ivar = np.load("%s/val_ivar_norm.npz" %DATA_DIR)['arr_0']
    val_label = np.load("%s/val_label.npz" %DATA_DIR)['arr_0']

    coeffs = np.load("%s/coeffs.npz" %DATA_DIR)['arr_0']
    scatters = np.load("%s/scatters.npz" %DATA_DIR)['arr_0']
    chisqs = np.load("%s/chisqs.npz" %DATA_DIR)['arr_0']
    pivots = np.load("%s/pivots.npz" %DATA_DIR)['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, val_label[:,0:4],
            val_id, val_flux, val_ivar)

    np.savez("%s/val_SNR.npz" %DATA_DIR, ds.test_SNR)

    ds.set_label_names(["Teff", "logg", "FeH", "aFe"])
    md = model.CannonModel(2)
    md.coeffs = coeffs
    md.scatters = scatters
    md.chisqs = chisqs
    md.pivots = pivots
    md.diagnostics_leading_coeffs(ds)

    nguesses = 7
    nobj = len(ds.test_ID)
    nlabels = ds.tr_label.shape[1]
    choose = np.random.randint(0,nobj,size=nguesses)
    starting_guesses = ds.tr_label[choose]-md.pivots
    labels = np.zeros((nguesses, nobj, nlabels))
    chisq = np.zeros((nguesses, nobj))
    errs = np.zeros(labels.shape)

    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,md,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    np.savez("%s/val_labels_all_starting_vals.npz" %DATA_DIR, labels)
    np.savez("%s/val_chisq_all_starting_vals.npz" %DATA_DIR, chisq)
    np.savez("%s/val_errs_all_starting_vals.npz" %DATA_DIR, errs)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("%s/val_cannon_labels.npz" %DATA_DIR, best_labels)
    np.savez("%s/val_errs.npz" %DATA_DIR, best_errs)
    np.savez("%s/val_chisq.npz" %DATA_DIR, best_chisq)

    ds.test_label_vals = best_labels
    ds.diagnostics_1to1()

def test():
    wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
    tr_id = np.load("%s/tr_id.npz" %DATA_DIR)['arr_0']
    tr_flux = np.load("%s/tr_flux_norm.npz" %DATA_DIR)['arr_0']
    tr_ivar = np.load("%s/tr_ivar_norm.npz" %DATA_DIR)['arr_0']
    test_id = np.load("%s/test_id.npz" %DATA_DIR)['arr_0']
    test_flux = np.load("%s/test_flux_norm.npz" %DATA_DIR)['arr_0']
    test_ivar = np.load("%s/test_ivar_norm.npz" %DATA_DIR)['arr_0']
    tr_label = np.load("%s/tr_label.npz" %DATA_DIR)['arr_0']

    coeffs = np.load("%s/coeffs.npz" %DATA_DIR)['arr_0']
    scatters = np.load("%s/scatters.npz" %DATA_DIR)['arr_0']
    chisqs = np.load("%s/chisqs.npz" %DATA_DIR)['arr_0']
    pivots = np.load("%s/pivots.npz" %DATA_DIR)['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label[:,0:4],
            test_id, test_flux, test_ivar)

    np.savez("%s/test_SNR.npz" %DATA_DIR, ds.test_SNR)

    ds.set_label_names(["Teff", "logg", "FeH", "aFe"])
    md = model.CannonModel(2)
    md.coeffs = coeffs
    md.scatters = scatters
    md.chisqs = chisqs
    md.pivots = pivots
    md.diagnostics_leading_coeffs(ds)

    nguesses = 7
    nobj = len(ds.test_ID)
    nlabels = ds.tr_label.shape[1]
    choose = np.random.randint(0,nobj,size=nguesses)
    starting_guesses = ds.tr_label[choose]-md.pivots
    labels = np.zeros((nguesses, nobj, nlabels))
    chisq = np.zeros((nguesses, nobj))
    errs = np.zeros(labels.shape)

    ds.tr_label = np.zeros((nobj, nlabels))
    
    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,md,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    np.savez("%s/labels_all_starting_vals.npz" %DATA_DIR, labels)
    np.savez("%s/chisq_all_starting_vals.npz" %DATA_DIR, chisq)
    np.savez("%s/errs_all_starting_vals.npz" %DATA_DIR, errs)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("%s/test_cannon_labels.npz" %DATA_DIR, best_labels)
    np.savez("%s/test_errs.npz" %DATA_DIR, best_errs)
    np.savez("%s/test_chisq.npz" %DATA_DIR, best_chisq)

    ds.test_label_vals = best_labels

if __name__=="__main__":
    #choose_reference_set()
    #normalize_ref_set()
    #normalize_test_set()
    #choose_training_set()
    #train()
    #validate()
    test()
