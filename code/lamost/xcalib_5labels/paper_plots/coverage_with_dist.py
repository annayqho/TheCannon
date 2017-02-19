# read in all LAMOST labels

import numpy as np
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def calc_dist(lamost_point, training_points, coeffs):
    """ avg dist from one lamost point to nearest 10 training points """
    diff2 = (training_points - lamost_point)**2
    dist = np.sqrt(np.sum(diff2*coeffs, axis=1))
    return np.mean(dist[dist.argsort()][0:10])


def dist(lamost_point, cannon_point, coeffs):
    diff2 = (lamost_point - cannon_point)**2
    dist = np.sqrt(np.sum(diff2*coeffs,axis=1))
    return dist


coeffs = 1./(np.array([100,0.2,0.1])**2)

# get all the training set values
with np.load("../../examples/test_training_overlap/tr_label.npz") as a:
    training_points = a['arr_0'][:,0:3]

direc = "../../examples/lamost_dr2"
teff_all = np.loadtxt(
        "%s/lamost_labels_all_dates.csv" %direc, 
        delimiter=',', dtype='float', usecols=(1,), 
        skiprows=1)

logg_all = np.loadtxt(
        "%s/lamost_labels_all_dates.csv" %direc, 
        delimiter=',', dtype='float', usecols=(2,), 
        skiprows=1)

feh_all = np.loadtxt(
        "%s/lamost_labels_all_dates.csv" %direc, 
        delimiter=',', dtype='float', usecols=(3,), 
        skiprows=1)

teff = np.loadtxt(
        "%s/lamost_labels_20121125.csv" %direc, delimiter=',', 
        dtype='float', usecols=(1,), skiprows=1)
logg = np.loadtxt(
        "%s/lamost_labels_20121125.csv" %direc, delimiter=',',         
        dtype='float', usecols=(2,), skiprows=1)
feh = np.loadtxt(
        "%s/lamost_labels_20121125.csv" %direc, delimiter=',',
        dtype='float', usecols=(3,), skiprows=1)

lamost_points = np.vstack((teff,logg,feh)).T

# calculate distances
training_dist = np.array(
        [calc_dist(p, training_points, coeffs) for p in lamost_points])

# plot all
plt.figure(figsize=(10,8))
plt.hist2d(teff_all,logg_all,bins=1000,norm=LogNorm(), cmap="Greys")
plt.ylim(1.5,5)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

# plot training set for training_dist < 2.5

cut = training_dist < 2.5
plt.scatter(teff[cut],logg[cut],c='darkorange',s=1,lw=0, 
            label=r"Distance from Training Labels $<$ 2.5")
cut = training_dist > 2.5
im = plt.scatter(teff[cut],logg[cut],c='darkorchid',s=1,lw=0, 
            label=r"Distance from Training Labels $>$ 2.5")
plt.legend(loc='upper left', fontsize=16, markerscale=5)
plt.xlabel(r"$\mbox{T}_{\mbox{eff}}$ [K] from LAMOST DR2", fontsize=16)
plt.ylabel(r"log g [dex] from LAMOST DR2", fontsize=16)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.savefig("ts_distance_in_full_lamost_label_space.png")
plt.close()
