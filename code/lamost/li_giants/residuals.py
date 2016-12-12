""" Calculate residuals """

import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
from matplotlib.ticker import MaxNLocator
import sys
sys.path.insert(0, '/home/annaho/TheCannon')
from TheCannon import model
from TheCannon import dataset
from plot_residual import plot


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
    m.coeffs = np.load(direc + "/coeffs.npz")['arr_0']
    m.scatters = np.load(direc + "/scatters.npz")['arr_0']
    m.chisqs = np.load(direc + "/chisqs.npz")['arr_0']
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
    DATA_DIR = "/home/annaho/TheCannon/data/lamost"
    WL_DIR = "/home/annaho/TheCannon/code/lamost/mass_age/cn"
    wl = np.load(WL_DIR + "/wl_cols.npz")['arr_0']
    ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
    test_label = np.load("%s/%s_all_cannon_labels.npz" %(DATA_DIR,date))['arr_0']
    ds.test_label_vals = test_label
    ds.test_flux = np.load("%s/%s_test_flux.npz" %(DATA_DIR,date))['arr_0']
    ds.test_ivar = np.load("%s/%s_test_ivar.npz" %(DATA_DIR,date))['arr_0']
    return ds


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
    resid = get_residuals(ds, m)
    plot(0, ds.wl, ds.test_flux, ds.test_ivar, m.
