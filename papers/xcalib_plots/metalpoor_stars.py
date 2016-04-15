import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from math import log10, floor
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

def round_sig(x, sig=2):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)

names = ['T_{eff},', '\log g', '[Fe/H]']
units = ['K', 'dex', 'dex']
snr_str = [r'SNR $\textless$ 50', r'50 $\textless$ SNR $\textless$ 100', r'SNR $\textgreater$ 100']
snr_str = snr_str[::-1]
cutoffs = [0, 50, 100, 10000]
cutoffs = cutoffs[::-1]
y_highs = [300, 0.5, 0.3]
x_lows = [4000, 1.1, -0.9, -0.08]
x_highs = [5300, 3.8, 0.4, 0.4]

all_cannon = np.load("run_2_train_on_good/optimization_experiment/best_labels.npz")['arr_0'].T
all_ids = np.load("run_2_train_on_good/all_ids.npz")['arr_0']
all_apogee = np.load("run_2_train_on_good/all_label.npz")['arr_0']
all_snr = np.load("run_2_train_on_good/SNRs.npz")['arr_0']
IDs_lamost = np.loadtxt(
        "../examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt",
        usecols=(0,), dtype=(str))
IDs_apogee = np.loadtxt(
        "../examples/apogee_sorted_by_ra.txt",
        usecols=(0,), dtype=(str))
labels_all_lamost = np.loadtxt(
        "../examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt",
        usecols=(3,4,5), dtype=(float))
inds = np.array([np.where(IDs_lamost==a)[0][0] for a in all_ids])
labels_lamost = labels_all_lamost[inds,:]
ids_apogee = IDs_apogee[inds]
labels_apogee = all_apogee[inds]

good = labels_apogee[:,2] > -500
metalpoor_stars = apogee[good][apogee_labels[:,2][good]<-1.8]

np.savetxt("metalpoor_stars.txt", metalpoor_stars)
#plt.hist2d(lamost[:,2][good], apogee[:,2][good], norm=LogNorm(), cmap="gray_r", bins=50)
#plt.xlabel("LAMOST [Fe/H] [dex]")
#plt.ylabel("APOGEE [Fe/H] [dex]")
#plt.title("Comparing [Fe/H] in LAMOST and APOGEE")
#plt.plot([-3,1], [-3,1])
#plt.savefig("metalpoor_stars.png")
#plt.close()


