import matplotlib
matplotlib.use("GtkAgg")
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np


direc = "../run_9b_reddening"

all_cannon = np.load("%s/all_cannon_labels.npz" %direc)['arr_0']
all_ids = np.load("../run_2_train_on_good/all_ids.npz")['arr_0']
all_apogee = np.load("../run_2_train_on_good/all_label.npz")['arr_0']
good_id = np.load("%s/tr_id.npz" %direc)['arr_0']
choose = np.array([np.where(all_ids==val)[0][0] for val in good_id])
apogee = all_apogee[choose]
cannon = all_cannon

IDs_lamost = np.loadtxt(
    "../../examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt",
    usecols=(0,), dtype=(str))
labels_all_lamost = np.loadtxt(
    "../../examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt",
    usecols=(3,4,5), dtype=(float))
inds = np.array([np.where(IDs_lamost==a)[0][0] for a in good_id])
lamost = labels_all_lamost[inds,:]

data = [lamost, cannon, apogee]

low = 3800
high = 5500

low2 = 0.5
high2 = 4.0

fig,axarr = plt.subplots(1,3, figsize=(10,5.5), sharex=True, sharey=True)

names = ['LAMOST DR2', 'Cannon/LAMOST', 'APOGEE DR12']

for i in range(0, len(names)):
    ax = axarr[i]
    use = data[i]
    im = ax.hist2d(use[:,0], use[:,1], norm=LogNorm(), bins=100, 
            cmap="inferno", range=[[low,high],[low2,high2]], vmin=1,vmax=70)
    ax.set_xlabel(r"$\mbox{T}_{\mbox{eff}}$" + " [K]", fontsize=16)
    if i == 0:
        ax.set_ylabel("log g [dex]", fontsize=16)
    ax.set_title("%s" %names[i], fontsize=16)
    ax.set_xlim(low,high)
    ax.set_ylim(low2,high2)
    ax.tick_params(axis='x', labelsize=16)
    ax.locator_params(nbins=5)
    #if i == 2: fig.colorbar(im[3], cax=ax, label="log(Number of Objects)")
    #plt.savefig("rc_%s.png" %names)
    #plt.close()

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = plt.colorbar(im[3], cax=cbar_ax)
cbar.set_label("log(density)", size=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.tick_params(labelsize=16)

plt.savefig("rc_5label.png")
