"""
In this run, I train TC on all *good* objects in the 11,057 overlap set 
(this is the 9594 objects from run_2, + the 500-something objects in 
examples/example_DR12/temp_keep_metalpoor.txt) and also discard items that were
deemed to be outliers during the take-none-out test.
Now, we're spicing things up by adding C and N as labels. Now, we have six labels. 
"""

import numpy as np
import glob
from astropy.table import Table
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/annaho/TheCannon')
sys.path.insert(0, '/home/annaho')
from TheCannon import dataset
from TheCannon import model
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import os


def load_data():
    # find the appropriate values
    table = Table.read("../run_13_all_abundances_xcalib/apogee_dr12_labels.csv")
    ids_all_apogee = table['APOGEE_ID']
    lamost_key = np.loadtxt("../../examples/lamost_sorted_by_ra.txt", dtype=str)
    apogee_key = np.loadtxt("../../examples/apogee_sorted_by_ra.txt", dtype=str)
    inds = []
    for val in ids_all_apogee:
        new_val = "aspcapStar-r5-v603-"+val+".fits"
        inds.append(np.where(apogee_key==new_val)[0][0])
    all_id = lamost_key[inds]

    all_label = np.vstack(
            (table['TEFF'], table['LOGG'],
             table['C_FE'], table['N_FE'], table['AL_FE'], table['CA_FE'], 
             table['FE_H'], table['K_FE'], table['MG_FE'], table['MN_FE'], 
             table['NA_FE'], table['NI_FE'], table['O_FE'], table['SI_FE'], 
             table['S_FE'], table['TI_FE'], table['V_FE'])).T
    
    id_key = np.load("../run_9_more_metal_poor/tr_id.npz")['arr_0']
    snr_key = np.load("../run_9_more_metal_poor/tr_snr.npz")['arr_0']
    snr = []
    for val in all_id:
        if val in id_key:
            ind = np.where(id_key==val)[0][0]
            snr.append(snr_key[ind])
        else:
            snr.append(0)
    pick = np.array(snr) > 0
    np.savez("all_id.npz", all_id[pick])
    np.savez("tr_id.npz", all_id[pick][0:])
    np.savez("all_label.npz", all_label[pick])
    np.savez("tr_label.npz", all_label[pick][0:])


def train():
    tr_id = np.load("tr_id.npz")['arr_0']
    tr_label = np.load("tr_label.npz")['arr_0']

    tr_flux = np.load("tr_flux.npz")['arr_0']
    tr_ivar = np.load("tr_ivar.npz")['arr_0']

    wl = np.load("./wl.npz")['arr_0']
    label_names = ['T_{eff}', '\log g', 'Al/Fe', 'Ca/Fe',
                   'C/Fe', 'Fe/H', 'K/Fe', 'Mg/Fe', 'Mn/Fe','Na/Fe', 'Ni/Fe','N/Fe', 'O/Fe', 
                   'Si/Fe', 'S/Fe', 'Ti/Fe', 'V/Fe']

    ds = dataset.Dataset(
        wl, tr_id, tr_flux, tr_ivar, tr_label, tr_id, tr_flux, tr_ivar)
    print("Training with %s Objects" %tr_flux.shape[0])

    ds.set_label_names(label_names)
    ds.diagnostics_SNR()
    #ds.diagnostics_ref_labels()
    np.savez("tr_snr.npz", ds.tr_SNR)

    m = model.CannonModel(2)
    m.fit(ds)
    np.savez("./coeffs.npz", m.coeffs)
    np.savez("./scatters.npz", m.scatters)
    np.savez("./chisqs.npz", m.chisqs)
    np.savez("./pivots.npz", m.pivots)
    m.diagnostics_leading_coeffs(ds)
    #m.diagnostics_leading_coeffs_triangle(ds)
    m.diagnostics_plot_chisq(ds)


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq


def test_step():
    wl = np.load("./wl.npz")['arr_0']
    tr_id = np.load("./tr_id.npz")['arr_0']
    tr_flux = np.load("./tr_flux.npz")['arr_0']
    tr_ivar = np.load("./tr_ivar.npz")['arr_0']
    tr_label = np.load("./tr_label.npz")['arr_0']

    ds = dataset.Dataset(wl, tr_id, tr_flux, tr_ivar, tr_label, tr_id, tr_flux, tr_ivar)
    label_names = ['T_{eff}', '\log g', 'Al/Fe', 'Ca/Fe',
                   'C/Fe', 'Fe/H', 'K/Fe', 'Mg/Fe', 'Mn/Fe','Na/Fe', 'Ni/Fe','N/Fe', 'O/Fe', 
                   'Si/Fe', 'S/Fe', 'Ti/Fe', 'V/Fe']
    ds.set_label_names(label_names)

    m = model.CannonModel(2)
    m.coeffs = np.load("./coeffs.npz")['arr_0']
    m.scatters = np.load("./scatters.npz")['arr_0']
    m.chisqs = np.load("./chisqs.npz")['arr_0']
    m.pivots = np.load("./pivots.npz")['arr_0']
    m.diagnostics_leading_coeffs(ds, figname="leading_coeffs_short.png")

    nguesses = 7
    nlabels = ds.tr_label.shape[1]
    nobj = ds.tr_label.shape[0]
    choose = np.random.randint(0,nobj,size=nguesses)
    starting_guesses = ds.tr_label[choose]-m.pivots

    labels = np.zeros((nguesses, nobj, nlabels)) # 4,10955,4
    #labels = np.zeros(starting_guesses.shape)
    chisq = np.zeros((nguesses, nobj))
    
    for ii,guess in enumerate(starting_guesses):
        a,b = test_step_iteration(ds,m,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b

    np.savez("labels_all_starting_vals.npz", labels)
    np.savez("chisq_all_starting_vals.npz", chisq)

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros(labels[0].shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]

    np.savez("./all_cannon_labels.npz", best_labels)
    np.savez("./cannon_label_chisq.npz", best_chisq)

    ds.test_label_vals = best_labels
    ds.diagnostics_survey_labels()
    ds.diagnostics_1to1(figname = "1to1_test_label")


if __name__=="__main__":
    #load_data()
    #train()
    print("test")
    test_step()
