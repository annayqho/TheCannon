"""
Divide 9952 training objects into eight groups, 
and do an 8-fold leave-1/8 out. 
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon/TheCannon')
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon')
from TheCannon import dataset
from TheCannon import model
from TheCannon import lamost
from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os
import pyfits

direc_ref = "/Users/annaho/TheCannon/data/lamost_paper"


def group_data():
    """ Load the reference data, and assign each object
    a random integer from 0 to 7. Save the IDs. """

    tr_obj = np.load("%s/ref_id.npz" %direc_ref)['arr_0']
    groups = np.random.randint(0, 8, size=len(tr_obj))
    np.savez("ref_groups.npz", groups)


def train(ds, ii):
    """ Run the training step, given a dataset object. """
    print("Loading model")
    m = model.CannonModel(2)
    print("Training...")
    m.fit(ds)
    np.savez("./ex%s_coeffs.npz" %ii, m.coeffs)
    np.savez("./ex%s_scatters.npz" %ii, m.scatters)
    np.savez("./ex%s_chisqs.npz" %ii, m.chisqs)
    np.savez("./ex%s_pivots.npz" %ii, m.pivots)
    fig = m.diagnostics_leading_coeffs(ds)
    plt.savefig("ex%s_leading_coeffs.png" %ii)
    # m.diagnostics_leading_coeffs_triangle(ds)
    # m.diagnostics_plot_chisq(ds)
    return m


def test(ds, m, group):
    nguesses = 7
    nobj = len(ds.test_ID)
    nlabels = len(m.pivots)
    choose = np.random.randint(0,nobj,size=nguesses)
    tr_label = ds.tr_label
    print("nlab")
    print(nlabels)
    print("nobj")
    print(nobj)
    print("tr label shape")
    print(tr_label.shape)
    print("m pivots shape")
    print(m.pivots.shape)
    starting_guesses = tr_label[choose]-m.pivots
    labels = np.zeros((nguesses, nobj, nlabels))
    chisq = np.zeros((nguesses, nobj))
    errs = np.zeros(labels.shape)

    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,m,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    np.savez("ex%s_labels_all_starting_vals.npz" %group, labels)
    np.savez("ex%s_chisq_all_starting_vals.npz" %group, chisq)
    np.savez("ex%s_errs_all_starting_vals.npz" %group, errs)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros(tr_label.shape)
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("./ex%s_cannon_label_vals.npz" %group, best_labels)
    np.savez("./ex%s_cannon_label_chisq.npz" %group, best_chisq)
    np.savez("./ex%s_cannon_label_errs.npz" %group, best_errs)

    ds.test_label_vals = best_labels
    ds.diagnostics_survey_labels()
    ds.diagnostics_1to1(figname = "ex%s_1to1_test_label" %group)


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def xvalidate():
    """ Train a model, leaving out a group corresponding
    to a random integer from 0 to 7, e.g. leave out 0. 
    Test on the remaining 1/8 of the sample. """

    print("Loading data")
    groups = np.load("ref_groups.npz")['arr_0']
    ref_label = np.load("%s/ref_label.npz" %direc_ref)['arr_0']
    ref_id = np.load("%s/ref_id.npz" %direc_ref)['arr_0']
    ref_flux = np.load("%s/ref_flux.npz" %direc_ref)['arr_0']
    ref_ivar = np.load("%s/ref_ivar.npz" %direc_ref)['arr_0']
    wl = np.load("%s/wl.npz" %direc_ref)['arr_0']

    num_models = 8

    for ii in np.arange(num_models):
        print("Leaving out group %s" %ii)
        train_on = groups != ii
        test_on = groups == ii

        tr_label = ref_label[train_on]
        tr_id = ref_id[train_on]
        tr_flux = ref_flux[train_on]
        tr_ivar = ref_ivar[train_on]
        print("Training on %s objects" %len(tr_id))
        test_label = ref_label[test_on]
        test_id = ref_id[test_on]
        test_flux = ref_flux[test_on]
        test_ivar = ref_ivar[test_on]
        print("Testing on %s objects" %len(test_id))

        print("Loading dataset...")
        ds = dataset.Dataset(
                wl, tr_id, tr_flux, tr_ivar, tr_label, 
                test_id, test_flux, test_ivar)
        ds.set_label_names(
                ['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]', 'AKWISE'])
        fig = ds.diagnostics_SNR()
        plt.savefig("ex%s_SNR.png" %ii)
        fig = ds.diagnostics_ref_labels()
        plt.savefig("ex%s_ref_label_triangle.png" %ii)
        np.savez("ex%s_tr_snr.npz" %ii, ds.tr_SNR)

        # train a model
        m = train(ds, ii)

        # test step
        ds.tr_label = test_label # to compare the results
        test(ds, m, ii)


if __name__=="__main__":
    # group_data()
    xvalidate()
