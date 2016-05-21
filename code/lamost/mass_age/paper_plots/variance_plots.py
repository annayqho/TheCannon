import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

errors = np.load("run_6_bootstrapping/label_variances.npz")['arr_0']
labels = np.load("run_5_train_on_good/all_cannon_labels.npz")['arr_0']

names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]']
units = ['K', 'dex', 'dex', 'dex']
mins = [0,0,0,0]
maxs = [20, 0.06, 0.03, 0.02]

fig, axarr = plt.subplots(2,2, sharex=True, sharey=True)

for i in range(0, len(names)):
    name = names[i]
    if i == 0:
        ax = axarr[0,0]
    elif i == 1:
        ax = axarr[0,1]
    elif i == 2:
        ax = axarr[1,0]
    elif i == 3:
        ax = axarr[1,1]
    im = ax.scatter(labels[:,0], labels[:,1], c=errors[i,:], marker='x', vmin=mins[i], vmax=maxs[i], alpha=0.5, cmap="winter")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r"Cannon-LAMOST %s (%s)" %(names[0], units[0]))
    ax.set_ylabel(r"Cannon-LAMOST %s (%s)" %(names[1], units[0]))
    fig.colorbar(im, ax=ax, label="Error in %s" %name)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.show()
