""" Calculate residuals """

import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
from matplotlib.ticker import MaxNLocator
from TheCannon import model
from TheCannon import dataset

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
    m.infer_spectra(ds)
    resid = ds.test_flux - m.model_spectra
    return resid


def load_model(direc):
    """ Load the model 

    Parameters
    ----------
    direc: directory with all of the model files
    
    Returns
    -------
    m: model object
    """
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
    ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
    test_label = np.load("%s/xval_cannon_label_vals.npz" %DATA_DIR)['arr_0']
    ds.test_label_vals = test_label
    tr_flux = np.load("%s/ref_flux.npz" %DATA_DIR)['arr_0']
    tr_ivar = np.load("%s/ref_ivar.npz" %DATA_DIR)['arr_0']
    ds.test_flux = tr_flux
    ds.test_ivar = tr_ivar
    test_label = np.load("%s/xval_cannon_label_vals.npz" %DATA_DIR)['arr_0']
    ds.test_label_vals = test_label
    tr_flux = np.load("%s/ref_flux.npz" %DATA_DIR)['arr_0']
    tr_ivar = np.load("%s/ref_ivar.npz" %DATA_DIR)['arr_0']
    ds.test_flux = tr_flux
    ds.test_ivar = tr_ivar


def plot_fit():
    plt.plot(wl, resid[ii])
    plt.xlim(6400,7000)
    plt.ylim(-0.1,0.1)
    plt.axvline(x=6707.8, c='r', linestyle='--', linewidth=2)
    plt.axvline(x=6103, c='r', linestyle='--', linewidth=2)
    plt.show()
    plt.savefig("resid_%s.png" %ii)
    plt.close()


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
    DIR = "/Users/annaho/Data/LAMOST/Label_Transfer"
    wl = np.load(DIR + "/wl.npz")['arr_0']
    ids = np.load(DIR + "/tr_id.npz")['arr_0']
    flux = np.load(DIR + "/tr_flux.npz")['arr_0']
    ivar = np.load(DIR + "/tr_ivar.npz")['arr_0']
    label = np.load(DIR + "/all_cannon_labels.npz")['arr_0']
    ds = dataset.Dataset(wl, [], [], [], [], ids, flux, ivar)
    ds.test_label_vals = label
    m = load_model(DIR)
    print(ds.test_flux.shape)
    print(m.coeffs.shape)
    resid = get_residuals(ds, m)
