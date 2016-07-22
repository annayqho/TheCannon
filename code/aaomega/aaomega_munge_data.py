""" Prepare AAOmega data for The Cannon """
import glob
import numpy as np
from astropy.table import Table
from astropy import stats
import pyfits


DATA_DIR = "/Users/annaho/Data/AAOmega/Run_13_July"


def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def estimate_noise(fluxes, contmask):
    """ Estimate the scatter in a region of the spectrum
    taken to be continuum """
    nstars = fluxes.shape[0]
    scatter = np.zeros(nstars)
    for i,spec in enumerate(fluxes): 
        cont = spec[contmask]
        scatter[i] = stats.funcs.mad_std(cont)
    return scatter


def load_ref_spectra():
    """ Pull out wl, flux, ivar from files of training spectra """
    data_dir = "/Users/annaho/Data/AAOmega/ref_spectra"
    # Load the files & count the number of training objects
    ff = glob.glob("%s/*.txt" %data_dir)
    nstars = len(ff)
    print("We have %s training objects" %nstars)
    
    # Read the first file to get the wavelength array
    f = ff[0]
    data = Table.read(f, format="ascii.fast_no_header")
    wl = data['col1']
    npix = len(wl)
    print("We have %s pixels" %npix)

    tr_flux = np.zeros((nstars,npix))
    tr_ivar = np.zeros(tr_flux.shape)

    for i,f in enumerate(ff):
        data = Table.read(f, format="ascii.fast_no_header")
        flux = data['col2']
        tr_flux[i,:] = flux
        sigma = data['col3']
        tr_ivar[i,:] = 1.0 / sigma**2

    return np.array(ff), wl, tr_flux, tr_ivar


def test_spectra_initial_cull():
    """ cull by radial velocity """
    ff_all = glob.glob("testspectra_new/*.fits")
    ff = []
    for f in ff_all:
        print(f)
        a = pyfits.open(f)
        vel = a[0].header['VHELIO']
        a.close()
        if np.abs(vel) < 500:
            ff.append(f)
    nobj = len(ff)
    np.savez("test_id.npz", ff)


def test_spectra_no_cull():
    """ Pull out test IDs of science spectra """
    data_dir = "/Users/annaho/Data/AAOmega/science_spectra"
    ff_all = glob.glob("%s/*.asc" %data_dir)
    np.savez("test_id.npz", ff_all)


def load_test_spectra():
    """ after you've done the initial cull, load the spectra """
    data_dir = "/Users/annaho/Data/AAOmega"
    ff = np.load("%s/test_id.npz" %data_dir)['arr_0']
    nobj = len(ff)
    wl = np.load("%s/wl.npz" %data_dir)['arr_0']
    npix = len(wl)
    test_flux = np.zeros((nobj, npix))
    for i,f in enumerate(ff):
        a = np.loadtxt("%s/%s" %(data_dir, f))
        test_flux[i,:] = a[:,1]
    return np.array(ff), test_flux


def load_labels():
    data_dir = "/Users/annaho/Data/AAOmega"
    data = Table.read("%s/asu.fit" %data_dir)
    field = data['Field']
    fib = data['Fib']
    ids = [(x+"."+str(y)).replace(" ", "") for x,y in zip(field,fib)]
    teff = data['Teff']
    logg = data['logg']
    mh = data['__m_H_']
    afe =data['__a_Fe_']
    vrot = data['Vrot']
    labels = np.vstack((teff,logg,mh,afe,vrot))
    return np.array(ids), labels.T


def load_data():
    data_dir = "/Users/annaho/Data/AAOmega"
    out_dir = "%s/%s" %(data_dir, "Run_13_July")

    """ Use all the above functions to set data up for The Cannon """
    ff, wl, tr_flux, tr_ivar = load_ref_spectra()

    """ pick one that doesn't have extra dead pixels """
    skylines = tr_ivar[4,:] # should be the same across all obj
    np.savez("%s/skylines.npz" %out_dir, skylines)

    contmask = np.load("%s/contmask_regions.npz" %data_dir)['arr_0']
    scatter = estimate_noise(tr_flux, contmask)
    ids, labels = load_labels()
    
    # Select the objects in the catalog corresponding to the files
    inds = []
    ff_short = []
    for fname in ff:
        val = fname.split("/")[-1]
        short = (val.split('.')[0] + '.' + val.split('.')[1])
        ff_short.append(short)
        if short in ids:
            ind = np.where(ids==short)[0][0]
            inds.append(ind)

    # choose the labels
    tr_id = ids[inds]
    tr_label = labels[inds]

    # find the corresponding spectra
    ff_short = np.array(ff_short)
    inds = np.array([np.where(ff_short==val)[0][0] for val in tr_id])
    tr_flux_choose = tr_flux[inds]
    tr_ivar_choose = tr_ivar[inds]
    scatter_choose = scatter[inds]
    np.savez("%s/wl.npz" %out_dir, wl)
    np.savez("%s/ref_id_all.npz" %out_dir, tr_id)
    np.savez("%s/ref_flux_all.npz" %out_dir, tr_flux_choose)
    np.savez("%s/ref_ivar_all.npz" %out_dir, tr_ivar_choose)
    np.savez("%s/ref_label_all.npz" %out_dir, tr_label)
    np.savez("%s/ref_spec_scat_all.npz" %out_dir, scatter_choose)

    # now, the test spectra
    test_id, test_flux = load_test_spectra()
    scatter = estimate_noise(test_flux, contmask) 
    np.savez("%s/test_id.npz" %out_dir, test_id)
    np.savez("%s/test_flux.npz" %out_dir, test_flux)
    np.savez("%s/test_spec_scat.npz" %out_dir, scatter)


def make_full_ivar():
    """ take the scatters and skylines and make final ivars """

    # skylines come as an ivar
    # don't use them for now, because I don't really trust them...
    # skylines = np.load("%s/skylines.npz" %DATA_DIR)['arr_0']

    ref_flux = np.load("%s/ref_flux_all.npz" %DATA_DIR)['arr_0']
    ref_scat = np.load("%s/ref_spec_scat_all.npz" %DATA_DIR)['arr_0']
    test_flux = np.load("%s/test_flux.npz" %DATA_DIR)['arr_0']
    test_scat = np.load("%s/test_spec_scat.npz" %DATA_DIR)['arr_0']
    ref_ivar = np.ones(ref_flux.shape) / ref_scat[:,None]**2
    test_ivar = np.ones(test_flux.shape) / test_scat[:,None]**2

    # ref_ivar = (ref_ivar_temp * skylines[None,:]) / (ref_ivar_temp + skylines)
    # test_ivar = (test_ivar_temp * skylines[None,:]) / (test_ivar_temp + skylines)

    ref_bad = np.logical_or(ref_flux <= 0, ref_flux > 1.1)
    test_bad = np.logical_or(test_flux <= 0, test_flux > 1.1)
    SMALL = 1.0 / 1000000000.0
    ref_ivar[ref_bad] = SMALL
    test_ivar[test_bad] = SMALL
    np.savez("%s/ref_ivar_corr.npz" %DATA_DIR, ref_ivar)
    np.savez("%s/test_ivar_corr.npz" %DATA_DIR, test_ivar)


if __name__=="__main__":
    #load_data()
    make_full_ivar()
