import numpy as np
import glob

cannon_teff = np.array([])
cannon_logg = np.array([])
cannon_feh = np.array([])
cannon_alpha = np.array([])

tr_teff = np.array([])
tr_logg = np.array([])
tr_feh = np.array([])
tr_alpha = np.array([])

a = glob.glob("./*tr_label.npz")
a.sort()

for filename in a:
    labels = np.load(filename)['arr_0']
    tr_teff = np.append(tr_teff, labels[:,0])
    tr_logg = np.append(tr_logg, labels[:,1])
    tr_feh = np.append(tr_feh, labels[:,2])
    tr_alpha = np.append(tr_alpha, labels[:,3])

a = glob.glob("./*cannon_labels.npz")
a.sort()

for filename in a:
    labels = np.load(filename)['arr_0']
    cannon_teff = np.append(cannon_teff, labels[:,0])
    cannon_logg = np.append(cannon_logg, labels[:,1])
    cannon_feh = np.append(cannon_feh, labels[:,2])
    cannon_alpha = np.append(cannon_alpha, labels[:,3])

a = glob.glob("./*_SNR.npz")
a.sort()

test_SNR = np.array([])

for filename in a:
    SNRs = np.load(filename)['arr_0']
    test_SNR = np.append(test_SNR, SNRs)

np.savez("test_SNR", test_SNR)
np.savez("tr_label", np.vstack((tr_teff, tr_logg, tr_feh, tr_alpha)))
np.savez("cannon_label", np.vstack((cannon_teff, cannon_logg, cannon_feh, cannon_alpha)))
