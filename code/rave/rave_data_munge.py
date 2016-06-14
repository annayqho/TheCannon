# Data munging for RAVE

# The Cannon needs: 
# length-L wavelength vec 
# an NxL block of training set pixel fluxes and corresponding ivars
# an NxK block of training label values
# an MxL block of test set pixel vals and corresponding ivars

def read_wl():
    filename = readsave('2013K1_parameters.save')
    items = inputf.items()
    data = items[0][1]
    wl = data['lambda'][0] # assuming they're all the same... 
    return wl


def read_tr_data():
    inputf = readsav('RAVE_DR4_calibration_data.save')
    items = inputf.items()
    data = items[0][1]
    tr_flux = data['spectrum'][0].T # shape (807, 839) = (nstars, npix)
    npix = tr_flux.shape[1]
    nstars = tr_flux.shape[0]
    teff = data['teff'][0] # length 807
    logg = data['logg'][0] # length 807
    feh = data['feh'][0]
    tr_label = np.vstack((teff, logg, feh)).T
    snr = np.zeros(nstars)
    snr.fill(100) # a guess for what the SNR could be
    tr_ivar = (snr[:,None]/tr_flux)**2
    return tr_flux, tr_ivar, tr_label


def read_test(filename):
    inputf = readsav(filename)
    items = inputf.items()
    data = items[0][1]
    sp = data['obs_sp'] # (75437, 839) 
    test_flux = np.zeros((len(sp), len(sp[0])))
    for jj in range(0, len(sp)):
        test_flux[jj,:] = sp[jj]
    snr = np.array(data['snr'])
    test_ivar = (snr[:,None]/test_flux)**2
    bad = np.logical_or(np.isnan(test_ivar), np.isnan(test_flux))
    test_ivar[bad] = 0.
    test_flux[bad] = 0.
    return (test_flux, test_ivar, wl)
