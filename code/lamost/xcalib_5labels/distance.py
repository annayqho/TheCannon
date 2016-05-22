def calc_dist(lamost_point, training_points, coeffs):
    """ dists from one lamost point to all training points """
    diff2 = (training_points - lamost_point)**2
    dist = np.sqrt(sum(diff2*coeffs, axis=1))
    return np.mean(dist[dist.argsort()][0:10])

# load data
import numpy as np

with np.load("test_training_overlap/tr_label.npz") as a: 
    training_points = a['arr_0'][:,0:3]

a = np.load("test_results.npz")
test_ID, test_labels = a['arr_0'], a['arr_1']
cannon_teff = test_labels[:,0]
cannon_logg = test_labels[:,1]
cannon_feh = test_labels[:,2]

label_file = "lamost_dr2/lamost_labels_20121125.csv"
lamost_label_id = np.loadtxt(label_file, usecols=(0,), dtype=str, delimiter=',', skiprows=1)
lamost_teff = np.loadtxt(label_file, usecols=(1,), dtype=float, delimiter=',', skiprows=1)
lamost_logg = np.loadtxt(label_file, usecols=(2,), dtype=float, delimiter=',', skiprows=1)
lamost_feh = np.loadtxt(label_file, usecols=(3,), dtype=float, delimiter=',', skiprows=1)

union = np.intersect1d(lamost_label_id, test_ID)
cannon_inds = [np.where(test_ID==item)[0][0] for item in union]
lamost_inds = [np.where(lamost_label_id==item)[0][0] for item in union]

cannon_points = np.vstack((cannon_teff[cannon_inds], cannon_logg[cannon_inds], cannon_feh[cannon_inds])).T
lamost_points = np.vstack((lamost_teff[lamost_inds], lamost_logg[lamost_inds], lamost_feh[lamost_inds])).T 

# the metric
coeffs = 1./(np.array([100,0.2,0.1])**2)
#1./(np.std(lamost_points, axis=0)**2)
# coeffs[1] = 1./coeffs[1]

# distance of each lamost point to training set
training_dist = np.array([calc_dist(point, training_points, coeffs) for point in lamost_points])
test_dist = np.sqrt(np.sum(coeffs*(cannon_points-lamost_points)**2, axis=1))

# scatter(training_dist, test_dist, c=lamost_teff[lamost_inds], marker='x', color='k')

rc('text', usetex=True)
rc('font', family='serif')
#scatter(lamost_teff[lamost_inds], lamost_logg[lamost_inds], marker='x', c=test_dist)
#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()
#xlabel(r"LAMOST $T_{eff}$")
#ylabel(r"LAMOST $\log g$")
#colorbar(label=r"$<\sqrt{C_{FeH}^{-2} \Delta^2 [Fe/H] + C_{Teff}^{-2} \Delta^2 T_{eff} + C_{logg}^{-2} \Delta^2 \log g}>$")
#title(r"$ \Delta$ (LAMOST, Cannon) in LAMOST Label Space for $[Fe/H] \textless -0.1$")
#
#region = np.logical_and(lamost_feh[lamost_inds]>-0.1, lamost_feh[lamost_inds]<0.1)
#region = np.logical_and(lamost_feh[lamost_inds]>-0.1, lamost_feh[lamost_inds]<0.1) 
#region = lamost_feh[lamost_inds] < -0.3 
#region = lamost_feh[lamost_inds] > 0.1
#scatter(lamost_teff[lamost_inds][region], lamost_logg[lamost_inds][region], marker='x', c=test_dist[region])
#savefig("distance_feh_low.png")
#

fig, axarr = subplots(2, sharex=True)
ax1 = axarr[0]
im1 = ax1.scatter(training_dist, test_dist, marker='x', c=lamost_teff[lamost_inds])
colorbar(im1, ax=ax1, label=r"LAMOST $T_{eff}$")
ax2 = axarr[1]
im2 = ax2.scatter(training_dist, test_dist, marker='x', c=lamost_logg[lamost_inds])
colorbar(im2, ax=ax2, label=r"LAMOST $\log g$")
ax2.set_xlabel("Dist from LAMOST point to avg(10 nearest training points)")

# investigating that weird grouping

scatter(training_dist, test_dist, marker='x', c= test_labels[:,2][cannon_inds])
colorbar(label=r"[Fe/H]")
scatter(training_dist, test_dist, marker='x', c= test_labels[:,3][cannon_inds])
colorbar(label=r"$\alpha/Fe")
