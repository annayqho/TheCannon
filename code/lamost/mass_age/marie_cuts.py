""" Apply cuts from Martig et al. 2016 """
from mass_age_functions import *

def get_mask(teff, logg, feh, cm, nm, afe):
    keep_teff = np.logical_and(teff > 4000, teff < 5000)
    keep_logg = np.logical_and(logg > 1.8, logg < 3.3)
    keep_feh = np.logical_and(feh > -0.8, feh < 0.25)
    keep_cm = np.logical_and(cm > -0.25, cm < 0.15)
    keep_nm = np.logical_and(nm > -0.1, nm < 0.45)
    keep_afe = np.logical_and(afe > -0.05, afe < 0.3)
    cplusn = calc_sum(feh, cm, nm)
    keep_cplusn = np.logical_and(cplusn > -0.1, cplusn < 0.15)
    cn = calc_cn(cm, nm)
    keep_cn = np.logical_and(cn > -0.6, cn < 0.2)
    temp = (np.vstack((
        keep_teff, keep_logg, keep_feh, keep_cm, keep_nm, 
        keep_afe, keep_cplusn, keep_cn))).astype(int)
    mask = np.sum(temp, axis=0)
    mask[mask < 8] = 0
    mask[mask == 8] = 1
    return mask.astype(bool)
    


