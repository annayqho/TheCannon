import numpy as np
import matplotlib.pyplot as plt


def asteroseismic_mass(nu_max, delta_nu, teff):
    teff_sol = 5777 # Kelvin
    nu_max_sol = 3140 # micro Hertz
    delta_nu_sol = 135.03 # micro Hertz
    M = (nu_max/nu_max_sol)**3 * (delta_nu/delta_nu_sol)**(-4) * (teff/teff_sol)**1.5
    return M


def calc_sum(mh, cm, nm):
    """ This is called [(C+N)/M] in Martig 2016 """
    C_H = cm+mh # computing [C/H]
    N_H = nm+mh # computing [N/H]
    Nc_Nh = 10 ** (C_H + 8.39 - 12)
    Nn_Nh = 10 ** (N_H + 7.78 - 12)
    Nc_Nh_sun = 10 ** (8.39 - 12)
    Nn_Nh_sun = 10 ** (7.78 - 12)
    CplusN = np.log10(Nc_Nh + Nn_Nh) - np.log10(Nc_Nh_sun + Nn_Nh_sun) - mh
    return CplusN

def calc_mass(mh,cm,nm):
    """ Table A1 in Martig 2016 """
    CplusN = calc_sum(mh,cm,nm)
    return (1.08 - 0.18*mh - 1.05*mh**2
            + 4.30*cm - 1.12*cm*mh - 49.92*cm**2
            + 1.43*nm - 0.67*nm * mh-41.04 * nm*cm - 0.63*nm**2
            - 7.55*CplusN - 1.30*CplusN *mh + 139.92*CplusN*cm + 47.33*CplusN*nm - 86.62*CplusN**2)

def calc_mass_2(mh,cm,nm,teff,logg):
    """ Table A2 in Martig 2016 """
    CplusN = calc_sum(mh,cm,nm)
    t = teff/4000.
    return (95.87 - 10.40*mh - 0.73*mh**2
            + 41.36*cm - 5.32*cm*mh - 46.78*cm**2
            + 15.05*nm - 0.93*nm*mh - 30.52*nm*cm - 1.61*nm**2
            - 67.61*CplusN + 7.05*CplusN*mh + 133.58*CplusN*cm + 38.94*CplusN*nm - 88.99*CplusN**2
            - 144.18*t + 5.12*t*mh - 73.77*t*cm - 15.29*t*nm + 101.75*t*CplusN + 27.77*t**2
            - 9.42*logg + 1.52*logg*mh + 16.04*logg*cm + 1.35*logg*nm - 18.65*logg*CplusN + 
            28.8*logg*t - 4.10*logg**2)


def calc_logAge(mh,cm,nm,teff,logg):
    """ Table A3 in Martig 2016 """
    CplusN = calc_sum(mh,cm,nm)
    t = teff/4000.
    return (-54.35 + 6.53*mh + 0.74*mh**2
            - 19.02*cm + 4.04*cm*mh + 26.90*cm**2
            - 12.18*nm + 0.76*nm*mh + 13.33*nm*cm - 1.04*nm**2
            + 37.22*CplusN - 4.94*CplusN*mh - 77.84*CplusN*cm - 17.60*CplusN*nm + 51.24*CplusN**2
            + 59.58*t-1.46*t*mh+48.29*t*cm+13.99*t*nm-65.67*t*CplusN+15.54*t**2
            + 16.14*logg-1.56*logg*mh-13.12*logg*cm-1.77*logg*nm+14.24*logg*CplusN-34.68*logg*t+
            4.17*logg**2)
