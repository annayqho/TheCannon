import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import pyfits
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


def test_step_iteration(ds, m, starting_guess):
    errs, chisq = m.infer_labels(ds, starting_guess)
    return ds.test_label_vals, chisq, errs


def create_sets(num_sets):
    ref_id = np.load("ref_id.npz")['arr_0']
    # Assign each object a number between 0 and num_sets
    nobj = len(ref_id)
    assignments = np.random.randint(num_sets, size=nobj)
    np.savez("assignments.npz", assignments)


def train(ds, leave_out):
    print("TRAINING")
    m = model.CannonModel(2)
    m.fit(ds)
    np.savez(
        "./model_%s.npz" %leave_out, 
        m.coeffs, m.scatters, m.chisqs, m.pivots) 
    fig = m.diagnostics_leading_coeffs(ds)
    plt.savefig("leading_coeffs_%s.png" %leave_out)
    plt.close()
    return m
    

def validate(ds, m, leave_out):
    print("VALIDATION")
    nguesses = 10
    ntestobj = len(ds.test_ID)
    print("%s test objects" %ntestobj)
    nlabels = len(m.pivots)
    choose = np.random.randint(0,ntestobj,size=nguesses)
    starting_guesses = ds.tr_label[choose]-m.pivots
    labels = np.zeros((nguesses, ntestobj, nlabels))
    chisq = np.zeros((nguesses, ntestobj))
    errs = np.zeros(labels.shape)

    for ii,guess in enumerate(starting_guesses):
        a,b,c = test_step_iteration(ds,m,starting_guesses[ii])
        labels[ii,:] = a
        chisq[ii,:] = b
        errs[ii,:] = c

    choose = np.argmin(chisq, axis=0)
    best_chisq = np.min(chisq, axis=0)
    best_labels = np.zeros(ds.tr_label.shape)
    best_errs = np.zeros(best_labels.shape)
    for jj,val in enumerate(choose):
        best_labels[jj,:] = labels[:,jj,:][val]
        best_errs[jj,:] = errs[:,jj,:][val]

    np.savez("test_results_%s.npz" %leave_out, best_labels, best_errs, best_chisq) 

    ds.test_label_vals = best_labels
    ds.diagnostics_1to1(figname="1to1_test_label_%s" %leave_out)


def loop(num_sets):
    wl = np.load("wl.npz")['arr_0']
    ref_id = np.load("ref_id.npz")['arr_0']
    ref_flux = np.load("ref_flux.npz")['arr_0']
    ref_ivar = np.load("ref_ivar.npz")['arr_0']
    ref_label = np.load("ref_label.npz")['arr_0']
    assignments = np.load("assignments.npz")['arr_0']
    
    print("looping through %s sets" %num_sets)
    for leave_out in range(1,num_sets):
        print("leaving out %s" %leave_out)
        training = assignments != leave_out
        test = assignments == leave_out
        tr_id = ref_id[training]
        tr_flux = ref_flux[training]
        tr_ivar = ref_ivar[training]
        tr_label = ref_label[training]
        np.savez(
            "tr_set_%s.npz" %leave_out, tr_id, tr_flux, tr_ivar, tr_label)
        test_id = ref_id[test]
        test_flux = ref_flux[test]
        test_ivar = ref_ivar[test]
        test_label = ref_label[test]
        np.savez(
            "test_set_%s.npz" %leave_out, test_id, test_flux, test_ivar, test_label)
        ds = dataset.Dataset(
            wl, tr_id, tr_flux, tr_ivar, tr_label, test_id, test_flux, test_ivar)
        ds.set_label_names(
            ['T_{eff}', '\log g', '[Fe/H]', '[C/M]','[N/M]', '[\\alpha/M]', 'A_k'])
        fig = ds.diagnostics_SNR()
        plt.savefig("SNRdist_%s.png" %leave_out)
        plt.close()
        fig = ds.diagnostics_ref_labels()
        plt.savefig("ref_label_triangle_%s.png" %leave_out)
        plt.close()
        np.savez("tr_snr_%s.npz" %leave_out, ds.tr_SNR)
        
        modelf = "model_%s.npz" %leave_out
        if glob.glob(modelf):
            print("model already exists")
            coeffs = np.load(modelf)['arr_0']
            scatters = np.load(modelf)['arr_1']
            chisqs = np.load(modelf)['arr_2']
            pivots = np.load(modelf)['arr_3']
            m = model.CannonModel(2)
            m.coeffs = coeffs
            m.scatters = scatters
            m.chisqs = chisqs
            m.pivots = pivots
        else:    
            m = train(ds, leave_out)
        ds.tr_label = test_label
        validate(ds, m, leave_out)

if __name__=="__main__":   
    #create_sets(8) 
    loop(8)
