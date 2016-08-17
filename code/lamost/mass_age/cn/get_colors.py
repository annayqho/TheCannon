import pyfits
import numpy as np

def get_colors(catalog):
    """ 
    Pull colors from catalog

    Parameters
    ----------
    catalog: filename
    """
    print("Get Colors")
    DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"
    a = pyfits.open(DATA_DIR + "/lamost_catalog_colors.fits")
    data = a[1].data
    a.close()
    all_ids = data['LAMOST_ID_1']
    all_ids = np.array([val.strip() for val in all_ids])
    # G magnitude
    gmag = data['gpmag']
    gmag_err = data['e_gpmag']
    # R magnitude
    rmag = data['rpmag']
    rmag_err = data['e_rpmag']
    # I magnitude
    imag = data['ipmag']
    imag_err = data['e_ipmag']
    # W1
    W1 = data['W1mag']
    W1_err = data['e_W1mag']
    # W1
    W2 = data['W2mag']
    W2_err = data['e_W2mag']
    # J magnitude
    Jmag = data['Jmag']
    Jmag_err = data['e_Jmag']
    # H magnitude
    Hmag = data['Hmag']
    Hmag_err = data['e_Hmag']
    # K magnitude
    Kmag = data['Kmag']
    Kmag_err = data['e_Kmag']
    # Stack
    mag = np.vstack((
        gmag, rmag, imag, Jmag, Hmag, Kmag, W2, W1)) # 8, nobj
    mag_err = np.vstack((
        gmag_err, rmag_err, imag_err, Jmag_err, 
        Hmag_err, Kmag_err, W2_err, W1_err))
    # Make g-r, r-i, i-J, etc
    col = mag[:-1] - mag[1:]
    col_ivar = 1/(mag_err[:-1]**2 + mag_err[1:]**2)

    # There's something wrong with the i-band, I think..so the second color r-i
    #bad = col[:,1] < 0.0
    #col_ivar[bad] = 0.0

    return col, col_ivar
