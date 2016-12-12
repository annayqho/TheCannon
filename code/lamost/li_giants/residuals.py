""" Calculate residuals """

from scipy.optimize import curve_fit
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
    return ds


def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0)
    return popt, pcov


def fit_li(ds, m, resid):
    scat = m.scatters
    iv_tot = (ds.test_ivar/(scat**2 * ds.test_ivar + 1))
    err = np.ones(iv_tot.shape)*1000
    err[iv_tot > 0] = 1/iv_tot[iv_tot>0]**0.5
    inds = np.logical_and(ds.wl >= 6700, ds.wl <= 6714)
    p0 = [-0.1, 6707, 1]
    popt, pcov = fit_gaussian(
            ds.wl[inds], resid[660,inds], yerr=err[660,inds], p0=p0)
    return popt, pcov


def run_all_data():
    """ Load the data that we're using to search for Li-rich giants.
    Store it in dataset and model objects. """
    DATA_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels"
    dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
    dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])
    for date in dates:
        print ("loading data for %s" %date)
        load_date(date)


if __name__=="__main__":
    # load a spectrum
    ds = load_dataset("20121006")
    m = load_model()
    model_spec = get_model_spectra(ds, m)
    resid = get_residuals(ds, m)
    popt, pcov = fit_li(ds, m, resid)
    #for ii in np.arange(len(ds.test_flux)):
    #for ii in np.arange(660, 661):
    #    plot(
    #            ii, ds.wl, ds.test_flux, ds.test_ivar, model_spec, 
    #            m.coeffs, m.scatters, m.chisqs, m.pivots)
