""" Prepare AAOmega data for The Cannon """
import glob
import numpy as np
from astropy.table import Table
from astropy import stats
import pyfits


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def estimate_noise(fluxes, contmask):
    """ Estimate the scatter around the mean in a region of the spectrum
    taken to be continuum """
    nstars = fluxes.shape[0]
    scatter = np.zeros(nstars)
    for i,spec in enumerate(fluxes): 
        cont = spec[contmask]
        scatter[i] = stats.funcs.mad_std(cont)
    return scatter


def load_spectra():
    # Load the files & count the number of training objects
    ff = glob.glob("training_spectra/*.txt")
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
    ff_all = glob.glob("science_spectra/*.asc")
    np.savez("test_id.npz", ff_all)


def load_test_spectra():
    """ after you've done the initial cull, load the spectra """
    ff = np.load("test_id.npz")['arr_0']
    nobj = len(ff)
    wl = np.load("wl.npz")['arr_0']
    npix = len(wl)
    test_flux = np.zeros((nobj, npix))
    for i,f in enumerate(ff):
        # if it's in a .fits file:
        #a = pyfits.open(f)
        #data = a[0].data
        #start_wl = a[0].header['CRVAL1']
        #delta_wl = a[0].header['CDELT1']
        #wl = start_wl + delta_wl * np.linspace(0,npix-1,npix)
        # if it's in a .txt file:
        a = np.loadtxt(f)
        test_flux[i,:] = a[:,1]

    return np.array(ff), test_flux


def load_labels():
    data = Table.read("asu.fit")
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


def load_raw_test_data():
    test_id, test_flux = load_test_spectra()
    contmask = np.load("contmask_regions.npz")['arr_0']
    scatter = estimate_noise(test_flux, contmask) 
    
    np.savez("test_id.npz", test_id)
    np.savez("test_flux.npz", test_flux)
    np.savez("test_spec_scat.npz", scatter)


def load_raw_data():
    ff, wl, tr_flux, tr_ivar = load_spectra()
    contmask = np.load("contmask_regions.npz")['arr_0']
    scatter = estimate_noise(tr_flux, contmask, tr_ivar) 
    ids, labels = load_labels()
    
    # Select the objects in the catalog corresponding to the files
    inds = []
    for val in ff:
        short = (val.split('.')[0] + '.' + val.split('.')[1]).split('/')[1]
        if short in ids:
            ind = np.where(ids==short)[0][0]
            inds.append(ind)

    # choose the labels
    tr_id = ids[inds]
    tr_label = labels[inds]

    # find the corresponding spectra
    ff_short = np.array([
            (val.split('.')[0] + '.' + val.split('.')[1]).split('/')[1] 
            for val in ff])

    inds = np.array([np.where(ff_short==val)[0][0] for val in tr_id])
    tr_flux_choose = tr_flux[inds]
    tr_ivar_choose = tr_ivar[inds]
    scatter_choose = scatter[inds]
    np.savez("wl.npz", wl)
    np.savez("id_all.npz", tr_id)
    np.savez("flux_all.npz", tr_flux_choose)
    np.savez("ivar_all.npz", tr_ivar_choose)
    np.savez("label_all.npz", tr_label)
    np.savez("spec_scat_all.npz", scatter_choose)


if __name__=="__main__":
    test_spectra_no_cull()
    load_raw_test_data()
