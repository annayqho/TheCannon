""" Calculate residuals """

from scipy.optimize import curve_fit
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
from matplotlib.ticker import MaxNLocator
from astropy.modeling import models, fitting
import sys
sys.path.insert(0, '/home/annaho/TheCannon')
from TheCannon import model
from TheCannon import dataset
from plot_residual import plot


def gaussian(x, a, b, c):
    val = a * np.exp(-(x-b)**2 / c**2)
    return val


def get_residuals(ds, m):
    """ Using the dataset and model object, calculate the residuals and return

    Parameters
    ----------
    ds: dataset object
    m: model object
    Return
    ------
    residuals: array of residuals, spec minus model spec
    """
    model_spectra = get_model_spectra(ds, m)
    resid = ds.test_flux - model_spectra
    return resid


def get_model_spectra(ds, m):
    m.infer_spectra(ds)
    return m.model_spectra


def load_model():
    """ Load the model 

    Parameters
    ----------
    direc: directory with all of the model files
    
    Returns
    -------
    m: model object
    """
    direc = "/home/annaho/TheCannon/code/lamost/mass_age/cn"
    m = model.CannonModel(2)
    m.coeffs = np.load(direc + "/coeffs.npz")['arr_0'][0:3626,:] # no cols
    m.scatters = np.load(direc + "/scatters.npz")['arr_0'][0:3626] # no cols
    m.chisqs = np.load(direc + "/chisqs.npz")['arr_0'][0:3626] # no cols
    m.pivots = np.load(direc + "/pivots.npz")['arr_0']
    return m


def load_dataset(date):
    """ Load the dataset for a single date 
    
    Parameters
    ----------
    date: the date (string) for which to load the data & dataset

    Returns
    -------
    ds: the dataset object
    """
    LAB_DIR = "/home/annaho/TheCannon/data/lamost"
    WL_DIR = "/home/annaho/TheCannon/code/lamost/mass_age/cn"
    SPEC_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels/output"
    wl = np.load(WL_DIR + "/wl_cols.npz")['arr_0'][0:3626] # no cols
    ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
    test_label = np.load("%s/%s_all_cannon_labels.npz" %(LAB_DIR,date))['arr_0']
    ds.test_label_vals = test_label
    a = np.load("%s/%s_norm.npz" %(SPEC_DIR,date))
    ds.test_flux = a['arr_0']
    ds.test_ivar = a['arr_1']
    ds.test_ID = np.load("%s/%s_ids.npz" %(SPEC_DIR,date))['arr_0']
    return ds


def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0)
    except RuntimeError:
        return 0,0
    return popt, pcov


def fit_li(x, y, yerr):
    p0 = [-0.1, 6707, 2]
    popt, pcov = fit_gaussian(
            x, y, yerr, p0)
    return popt, pcov


def get_data_to_fit(ii, ds, m, resid):
    scat = m.scatters
    iv_tot = (ds.test_ivar/(scat**2 * ds.test_ivar + 1))
    err = np.ones(iv_tot.shape)*1000
    err[iv_tot > 0] = 1/iv_tot[iv_tot>0]**0.5
    inds = np.logical_and(ds.wl >= 6700, ds.wl <= 6720)
    x = ds.wl[inds]
    y = resid[ii,inds]
    yerr=err[ii,inds]
    return x, y, yerr


def plot_fit(fit, x, y, yerr, figname='fit.png'):
    popt, pcov = fit
    plt.errorbar(x, y, yerr=yerr, fmt='.', c='k')
    xpts = np.linspace(min(x), max(x), 1000)
    plt.plot(xpts, gaussian(xpts, popt[0], popt[1], popt[2]), c='r', lw=2)
    plt.xlabel("Wavelength (Angstroms)", fontsize=16)
    plt.ylabel("Normalized Flux", fontsize=16)
    plt.savefig(figname)
    plt.close()
     

def select(yerrs, amps, amp_errs, widths):
    """ criteria for keeping an object """
    keep_1 = np.logical_and(amps < 0, widths > 1)
    keep_2 = np.logical_and(np.abs(amps) > 3*yerrs, amp_errs < 3*np.abs(amps))
    keep = np.logical_and(keep_1, keep_2)
    return keep



def get_name(filename):
    temp = filename.split('/')[-1]
    return temp.split('.')[0]



def run_one_date(date):
    # load a spectrum
    ds = load_dataset(date)
    nobj = len(ds.test_ID)
    m = load_model()
    model_spec = get_model_spectra(ds, m)
    resid = get_residuals(ds, m)

    dat = np.array([get_data_to_fit(i,ds,m,resid) for i in np.arange(nobj)])
    x = dat[:,0,:]
    y = dat[:,1,:]
    yerr = dat[:,2,:]
    med_err = np.median(yerr, axis=1)
    fits = np.array([fit_li(xval,yval,yerrval) for xval,yval,yerrval in zip(x,y,yerr)])
    popt = fits[:,0]
    pcov = fits[:,1]
    amps = 

    #     if fit == 0:
    #         amp = 999
    #         amp_err = 999
    #         width = 999
    #     else:
    #         amp = fit[0][0]
    #         amp_err = fit[1][0,0]
    #         width = fit[0][2]
    #     widths[ii] = width
    #     amps[ii] = amp
    #     amp_errs[ii] = amp_err
    #     if select(np.median(yerr), amp, amp_err, width):
    #         name = get_name(ds.test_ID[ii])
    #         li_rich_candidates.append(name)
    #         plot_fit(fit, x, y, yerr, figname="%s_%s_fit.png" %(date,name))
    #         plot(
    #                 ii, ds.wl, ds.test_flux, ds.test_ivar, model_spec,
    #                 m.coeffs, m.scatters, m.chisqs, m.pivots, 
    #                 figname="%s_%s_spec.png" %(date,name))
    # outf = open('%s_candidates.txt' %date, 'w')
    # outf.write("%s Candidates Total\n" %len(ds.test_ID))
    # for val in li_rich_candidates: outf.write("%s.fits\n" %val)
    # outf.close()


def run_all():
    """ Load the data that we're using to search for Li-rich giants.
    Store it in dataset and model objects. """
    DATA_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels"
    dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])

    for date in dates:
        if glob.glob("*%s*.txt" %date):
            print("%s done" %date)
        else:
            print("running %s" %date)
            run_one_date(date)
