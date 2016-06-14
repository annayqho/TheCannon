import numpy as np

a = open("all_flag_info.csv", "r")
lines = a.readlines()
a.close()

all_ids = np.load("run_2_train_on_good/all_ids.npz")['arr_0']
all_apogee = np.load("run_2_train_on_good/all_label.npz")['arr_0']

bad = []

# 3500 < Teff < 6000 # this is just a warning based on the range used in the synthetic grid
warn_teff = np.logical_or(all_apogee[:,0] < 3500, all_apogee[:,0] > 6000) # 112 stars
bad.extend(all_ids[warn_teff])

# logg < 0 & logg > 4, although we don't have any > 4
bad_logg = all_apogee[:,1] < 0
bad.extend(all_ids[bad_logg])
bad_logg = all_apogee[:,1] < 0 # 105 stars

# STAR_BAD: TEFF_BAD LOGG_BAD, X2_BAD, COLORTE_BAD, ROTATION_BAD, SN_BAD # 3 stars
for line in lines:
    if "STAR_BAD" in line:
        bad.append(line.split(',')[0])

# M_H_BAD or M_H_WARN # 456 stars
for line in lines:
    if np.logical_or("M_H_BAD" in line, "M_H_WARN" in line):
        bad.append(line.split(',')[0])

# ALPHA_M_BAD or ALPHA_M_WARN # 1 star
for line in lines:
    if np.logical_or("ALPHA_M_BAD" in line, "ALPHA_M_WARN" in line):
        bad.append(line.split(',')[0])

# translate from APOGEE to LAMOST ID
apogee = np.loadtxt("../examples/apogee_sorted_by_ra.txt", dtype=str)
lamost = np.loadtxt("../examples/lamost_sorted_by_ra.txt", dtype=str)
for ii,b in enumerate(bad):
    if b[0:4] != 'spec':
        ind = np.where(apogee == 'aspcapStar-r5-v603-' + b + '.fits')[0][0]
        bad[ii] = lamost[ind]

np.savez("bad_apogee_labels.npz", bad)
