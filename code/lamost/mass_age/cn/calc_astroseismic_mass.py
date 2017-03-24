""" Calculate mass using asteroseismic scaling relations """

def calc_mass(nu_max, delta_nu, teff):
    """ asteroseismic scaling relations """
    NU_MAX = 3140.0 # microHz
    DELTA_NU = 135.03 # microHz
    TEFF = 5777.0
    return (nu_max/NU_MAX)**3 * (delta_nu/DELTA_NU)**(-4) * (teff/TEFF)**1.5
