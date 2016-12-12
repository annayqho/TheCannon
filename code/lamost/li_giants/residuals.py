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
    ds.test_ID = np.load("%s/%s_ids.npz" %(SPEC_DIR,date))['arr_0']
    return ds


def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        fit = curve_fit(gaussian, x, y, sigma=yerr, p0=p0)
    except RuntimeError:
        return 0
    return fit


def fit_li(x, y, yerr):
    p0 = [-0.1, 6707, 1]
    fit = fit_gaussian(
            x, y, yerr, p0)
    return fit


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
    date = '20121006'
    ds = load_dataset(date)
    amps = np.zeros(len(ds.test_ID))
    amp_errs = np.zeros(amps.shape)
    m = load_model()
    model_spec = get_model_spectra(ds, m)
    resid = get_residuals(ds, m)
    for ii in np.arange(len(amps)):
        print(ii)
        x, y, yerr = get_data_to_fit(ii, ds, m, resid)
        fit = fit_li(x, y, yerr)
        if fit == 0:
            amp = 999
            amp_err = 999
        else:
            amp = fit[0][0]
            amp_err = fit[1][0,0]
        amps[ii] = amp
        amp_errs[ii] = amp_err
    np.savez("%s_fit_amplitudes.npz" %date, ds.test_ID, amps, amp_err)
    #plot_fit(fit, x, y, yerr)
    #for ii in np.arange(len(ds.test_flux)):
    #for ii in np.arange(660, 661):
    #    plot(
    #            ii, ds.wl, ds.test_flux, ds.test_ivar, model_spec, 
    #            m.coeffs, m.scatters, m.chisqs, m.pivots)
