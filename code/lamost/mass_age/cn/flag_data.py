""" Flag data that lies outside the zone of Marie's paper """

import numpy as np
import sys
sys.path.append('..')
from mass_age_functions import *

id_all = np.load("id_all.npz")['arr_0']
nobj = len(id_all)

flags = np.ones(nobj) # start with all flagged

# [M/H] > -0.8
feh = np.load("feh_all.npz")['arr_0']
good_feh = feh > -0.8

# 4000 < Teff < 5000
teff = np.load("teff_all.npz")['arr_0']
good_teff = np.logical_and(teff>4000, teff<5000)

# 1.8 < logg < 3.3
logg = np.load("logg_all.npz")['arr_0']
good_logg = np.logical_and(logg>1.8, logg<3.3)

# -0.25 < [C/M] < 0.15
cm = np.load("cm_all.npz")['arr_0']
good_cm = np.logical_and(cm > -0.25, cm < 0.15)

# -0.1 < [N/M] < 0.45
nm = np.load("nm_all.npz")['arr_0']
good_nm = np.logical_and(nm > -0.1, nm < 0.45)

# -0.1 < [(C+N)/M] < 0.15
cplusm = calc_sum(feh, cm, nm)
good_cplusm = np.logical_and(cplusm > -0.1, cplusm < 0.15)

# -0.6 < [C/N] < 0.2
cn = calc_cn(cm, nm)
good_cn = np.logical_and(cn > -0.6, cn < 0.2)

keep = np.logical_and.reduce(
        (good_feh, good_teff, good_logg, 
        good_cm, good_nm, good_cplusm, good_cn))

flags[keep] = 0

np.savez("flags.npz", flags)
