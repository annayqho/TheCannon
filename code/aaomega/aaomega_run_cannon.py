""" Apply The Cannon to the AAOmega Spectra! """

import numpy as np
import matplotlib.pyplot as plt
from TheCannon import dataset
from TheCannon import model


SMALL = 1.0 / 1000000000.0


def test_step_iteration(ds, md, starting_guess):
    errs, chisq = md.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def choose_reference_set():
    wl = np.load("wl.npz")['arr_0']
    all_id = np.load("id_all.npz")['arr_0']
    all_flux = np.load("flux_all.npz")['arr_0']
    all_scat = np.load("spec_scat_all.npz")['arr_0']
    # all_ivar = np.load("ivar_all.npz")['arr_0']
    all_ivar = np.ones(all_flux.shape) / all_scat[:,None]**2
    bad = np.logical_or(all_flux <= 0, all_flux > 1.1)
    all_ivar[bad] = SMALL
    np.savez("my_ivar_all.npz", all_ivar)
    all_label = np.load("label_all.npz")['arr_0']

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

    np.savez("ref_id.npz", ref_id)
    np.savez("ref_flux.npz", ref_flux)
    np.savez("ref_ivar.npz", ref_ivar)
    np.savez("ref_label.npz", ref_label)


def normalize_ref_set():
    wl = np.load("wl.npz")['arr_0']
    ref_id = np.load("ref_id.npz")['arr_0']
    ref_flux = np.load("ref_flux.npz")['arr_0']
    ref_ivar = np.load("ref_ivar.npz")['arr_0']
    ref_label = np.load("ref_label.npz")['arr_0']

    # contpix = np.load("wl_contpix_old.npz")['arr_0']
    # this array is a bit too long, clip it off
    # contpix_new = contpix[np.logical_and(contpix>8420, contpix<8700)]

    # inds = np.zeros(contpix_new.shape, dtype=int)
    # for i,val in enumerate(contpix_new):
        # find the nearest pixel
    #    inds[i] = int(np.argmin(np.abs(wl-val)))
        
    #contmask = np.zeros(len(wl), dtype=bool)
    #contmask[inds] = 1
    #np.savez("wl_contmask.npz", contmask)
    #print("SAVED")
    contmask = np.load("wl_contmask.npz")['arr_0']
    
    ds = dataset.Dataset(
            wl, ref_id, ref_flux, ref_ivar, ref_label, 
            ref_id, ref_flux, ref_ivar)
    ds.set_continuum(contmask)

    cont = ds.fit_continuum(3, "sinusoid")
    np.savez("ref_cont.npz", cont)
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
            ds.continuum_normalize(cont)
    bad = np.logical_or(norm_tr_flux <= 0, norm_tr_flux > 1.1)
    norm_tr_ivar[bad] = 0.0
    np.savez("ref_flux_norm.npz", norm_tr_flux)
    np.savez("ref_ivar_norm.npz", norm_tr_ivar)


def normalize_test_set():
    wl = np.load("wl.npz")['arr_0']
    test_id = np.load("test_id.npz")['arr_0']
    test_flux = np.load("test_flux.npz")['arr_0']
    test_scat = np.load("test_spec_scat.npz")['arr_0']
    test_ivar = np.ones(test_flux.shape) / test_scat[:,None]**2

    # for bad pixels (flux <= 0), set the ivar = 0
    # bad = np.logical_or(test_flux <= 0, test_flux >= 1.2)
    bad = test_flux <= 0
    test_ivar[bad] = SMALL
    np.savez("test_ivar.npz", test_ivar)

    contmask = np.load("wl_contmask.npz")['arr_0']

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
    np.savez("test_cont.npz", cont)
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
            ds.continuum_normalize(cont)
    # bad = np.logical_or(norm_test_flux <= 0, norm_test_flux >= 1.2)
    bad = norm_test_flux <= 0
    norm_test_ivar[bad] = 0.0
    np.savez("test_flux_norm.npz", norm_test_flux)
    np.savez("test_ivar_norm.npz", norm_test_ivar)


def choose_training_set():
    ref_id = np.load("ref_id.npz")['arr_0']
    ref_flux = np.load("ref_flux_norm.npz")['arr_0']
    ref_ivar = np.load("ref_ivar_norm.npz")['arr_0']
    ref_label = np.load("ref_label.npz")['arr_0']
    
    # randomly pick 80% of the objects to be the training set
    nobj = len(ref_id)
    assignments = np.random.randint(10, size=nobj)
    # if you're < 8, you're training
    choose = assignments < 8
    tr_id = ref_id[choose]
    tr_flux = ref_flux[choose]
    tr_ivar = ref_ivar[choose]
    tr_label = ref_label[choose]
    np.savez("tr_id.npz", tr_id)
    np.savez("tr_flux_norm.npz", tr_flux)
    np.savez("tr_ivar_norm.npz", tr_ivar)
    np.savez("tr_label.npz", tr_label)

    val_id = ref_id[~choose]
    val_flux = ref_flux[~choose]
    val_ivar = ref_ivar[~choose]
    val_label = ref_label[~choose]
    np.savez("val_id.npz", val_id)
    np.savez("val_flux_norm.npz", val_flux)
    np.savez("val_ivar_norm.npz", val_ivar)
    np.savez("val_label.npz", val_label)


def train():
    wl = np.load("wl.npz")['arr_0']
    tr_id = np.load("tr_id.npz")['arr_0']
    tr_flux = np.load("tr_flux_norm.npz")['arr_0']
    tr_ivar = np.load("tr_ivar_norm.npz")['arr_0']
    tr_label = np.load("tr_label.npz")['arr_0']
    val_id = np.load("val_id.npz")['arr_0']
    val_flux = np.load("val_flux_norm.npz")['arr_0']
    val_ivar = np.load("val_ivar_norm.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label[:,0:3], 
            val_id, val_flux, val_ivar)
    ds.set_label_names(["Teff", "logg", "FeH"])

    np.savez("tr_SNR.npz", ds.tr_SNR)

    fig = ds.diagnostics_SNR()
    plt.savefig("SNR_dist.png")
    plt.close()

    fig = ds.diagnostics_ref_labels()
    plt.savefig("ref_label_triangle.png")
    plt.close()

    md = model.CannonModel(2)
    md.fit(ds)

    fig = md.diagnostics_leading_coeffs(ds)
    plt.savefig("leading_coeffs.png")
    plt.close()

    np.savez("coeffs.npz", md.coeffs)
    np.savez("scatters.npz", md.scatters)
    np.savez("chisqs.npz", md.chisqs)
    np.savez("pivots.npz", md.pivots)


def validate():
    wl = np.load("wl.npz")['arr_0']
    tr_id = np.load("tr_id.npz")['arr_0']
    tr_flux = np.load("tr_flux_norm.npz")['arr_0']
    tr_ivar = np.load("tr_ivar_norm.npz")['arr_0']
    val_id = np.load("val_id.npz")['arr_0']
    val_flux = np.load("val_flux_norm.npz")['arr_0']
    val_ivar = np.load("val_ivar_norm.npz")['arr_0']
    val_label = np.load("val_label.npz")['arr_0']

    coeffs = np.load("coeffs.npz")['arr_0']
    scatters = np.load("scatters.npz")['arr_0']
    chisqs = np.load("chisqs.npz")['arr_0']
    pivots = np.load("pivots.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, val_label[:,0:3],
            val_id, val_flux, val_ivar)

    np.savez("val_SNR.npz", ds.test_SNR)

    ds.set_label_names(["Teff", "logg", "FeH"])
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

    np.savez("val_labels_all_starting_vals.npz", labels)
    np.savez("val_chisq_all_starting_vals.npz", chisq)
    np.savez("val_errs_all_starting_vals.npz", errs)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("val_cannon_labels.npz", best_labels)
    np.savez("val_errs.npz", best_errs)
    np.savez("val_chisq.npz", best_chisq)

    ds.test_label_vals = best_labels
    ds.diagnostics_1to1()

def test():
    wl = np.load("wl.npz")['arr_0']
    tr_id = np.load("tr_id.npz")['arr_0']
    tr_flux = np.load("tr_flux_norm.npz")['arr_0']
    tr_ivar = np.load("tr_ivar_norm.npz")['arr_0']
    test_id = np.load("test_id.npz")['arr_0']
    test_flux = np.load("test_flux_norm.npz")['arr_0']
    test_ivar = np.load("test_ivar_norm.npz")['arr_0']
    tr_label = np.load("tr_label.npz")['arr_0']

    coeffs = np.load("Model/coeffs.npz")['arr_0']
    scatters = np.load("Model/scatters.npz")['arr_0']
    chisqs = np.load("Model/chisqs.npz")['arr_0']
    pivots = np.load("Model/pivots.npz")['arr_0']

    ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label[:,0:3],
            test_id, test_flux, test_ivar)

    np.savez("test_SNR.npz", ds.test_SNR)

    ds.set_label_names(["Teff", "logg", "FeH"])
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

    np.savez("labels_all_starting_vals.npz", labels)
    np.savez("chisq_all_starting_vals.npz", chisq)
    np.savez("errs_all_starting_vals.npz", errs)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros((nobj, nlabels))
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("test_cannon_labels.npz", best_labels)
    np.savez("test_errs.npz", best_errs)
    np.savez("test_chisq.npz", best_chisq)

    ds.test_label_vals = best_labels

if __name__=="__main__":
    # choose_reference_set()
    # normalize_ref_set()
    # choose_training_set()
    # train()
    normalize_test_set()
    # validate()
    test()
