""" Estimate the uncertainty in the age measurement.

Sample from [Fe/H], [C/M], [N/M], Teff, logg to estimate
the width of the age measurement.
"""

import numpy as np
from matplotlib.pyplot as plt
from mass_age_functions import *

feh = np.load("feh_all.npz")['arr_0']
feh_scatter = 
cm = np.load("cm_all.npz")['arr_0']
nm = np.load("nm_all.npz")['arr_0']
teff = np.load("teff_all.npz")['arr_0']
logg = np.load("logg_all.npz")['arr_0']

calc_logAge(feh, cm, nm, teff, logg)
