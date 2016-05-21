# read in all LAMOST labels

import numpy as np
from matplotlib import rc
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

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
with np.load("../run_9b_reddening/tr_label.npz") as a:
    training_points = a['arr_0'][:,0:3]

label_file = "../../examples/lamost_dr2/lamost_labels_20121125.csv"
lamost_label_id = np.loadtxt(label_file, usecols=(0,), dtype=str, delimiter=',', skiprows=1)
lamost_teff = np.loadtxt(label_file, usecols=(1,), dtype=float, delimiter=',', skiprows=1)
lamost_logg = np.loadtxt(label_file, usecols=(2,), dtype=float, delimiter=',', skiprows=1)
lamost_feh = np.loadtxt(label_file, usecols=(3,), dtype=float, delimiter=',', skiprows=1)


a = np.load("../../examples/test_small_random/test_results.npz")
test_ID, test_labels = a['arr_0'], a['arr_1']
cannon_teff = test_labels[:,0]
cannon_logg = test_labels[:,1]
cannon_feh = test_labels[:,2]
cannon_alpha = test_labels[:,3]

union = np.intersect1d(lamost_label_id, test_ID)
cannon_inds = [np.where(test_ID==item)[0][0] for item in union]
lamost_inds = [np.where(lamost_label_id==item)[0][0] for item in union]


lamost_points = np.vstack((lamost_teff[lamost_inds],lamost_logg[lamost_inds],lamost_feh[lamost_inds])).T
cannon_points = np.vstack((cannon_teff[cannon_inds], cannon_logg[cannon_inds], cannon_feh[cannon_inds])).T

# calculate distances
training_dist = np.array([calc_dist(p, training_points, coeffs) for p in lamost_points])

diff2 = (lamost_points - cannon_points)**2
point_dist = np.sqrt(np.sum(diff2*coeffs,axis=1))

fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(2,1)

ax0 = plt.subplot(gs[0])
pl = ax0.hist(training_dist[training_dist < 15], bins=80, alpha=0.5, color='k', histtype='stepfilled')
ax0.set_ylabel("Number of Objects", fontsize=16)
ax0.tick_params(axis='x', labelsize=16)
ax0.tick_params(axis='y', labelsize=16)

ax1 = plt.subplot(gs[1], sharex=ax0)
cmap = cm.inferno
pl = ax1.scatter(training_dist,point_dist,c= lamost_logg[lamost_inds], marker='x', cmap='inferno', alpha=0.7)
cb = fig.colorbar(pl, orientation='horizontal')
cb.ax.tick_params(labelsize=16)
cb.set_label(label="log g [dex] from LAMOST", size=16)
ax1.set_xlim(-1, 10)
ax1.set_ylim(-1,15)
ax1.set_xlabel(r"Label Distance to Reference Set", fontsize=16)
ax1.set_ylabel(r"Label Distance: Cannon-LAMOST", fontsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

#plt.show()
plt.savefig("distance_cut.png")
