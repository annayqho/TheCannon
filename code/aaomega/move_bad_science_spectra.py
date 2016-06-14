import os
import numpy as np

# first criterion for badness: no flux whatsoever!

test_id = np.load("test_id.npz")['arr_0']
test_flux = np.load("test_flux.npz")['arr_0']
bad = np.median(test_flux, axis=1) == 0
bad_id = test_id[bad]

for fname in bad_id:
    os.system("mv %s science_spectra/no_flux" %fname)
