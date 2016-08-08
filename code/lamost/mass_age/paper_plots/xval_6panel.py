import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

def round_sig(x, sig=2):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)

DATA_DIR = "../cn"

names = ['T_{eff}', '\log g', '[M/H]', '[\\alpha/M]', '[C/M]', '[N/M]']
units = ['K', 'dex', 'dex', 'dex', 'dex', 'dex']
mins = [3700, 1.0, -1.0, -0.05, -0.3, -0.4]
maxs = [5300, 4.1, 0.6, 0.30, 0.2, 0.5]

cannon = np.load("%s/xval_cannon_label_vals.npz" %DATA_DIR)['arr_0']
all_id = np.load("%s/ref_id.npz" %DATA_DIR)['arr_0']
apogee = np.load("%s/ref_label.npz" %DATA_DIR)['arr_0']
snr = np.load("%s/ref_snr.npz" %DATA_DIR)['arr_0']

fig = plt.figure(figsize=(10,13))
gs = gridspec.GridSpec(3,2, wspace=0.3, hspace=0.3)

for i in range(0, len(names)):
    name = names[i]
    unit = units[i]
    low = mins[i]
    high = maxs[i]
    #low = np.minimum(min(apogee[:,i]), min(cannon[:,i]))
    #high = np.maximum(max(apogee[:,i]), max(cannon[:,i]))
    ax = plt.subplot(gs[i])
    ax.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
    #ax.legend(fontsize=14)
    bias = round_sig(np.mean(cannon[:,i]-apogee[:,i]), 3)
    scatter = round_sig(np.std(cannon[:,i]-apogee[:,i]), 3)
    ax.hist2d(
            apogee[:,i], cannon[:,i], range=[[low,high],[low,high]], 
            bins=50, norm=LogNorm(), cmap="gray_r")
    text = "bias = %s \nscatter = %s" %(bias, scatter)
    ax.text(
            0.05, 0.90, text, horizontalalignment='left', 
            verticalalignment='top', transform=ax.transAxes, fontsize=14)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r"$%s$" %name + " (%s) from APOGEE" %unit)
    ax.set_ylabel(r"$%s$" %(name) + " (%s) from Cannon/LAMOST" %unit)

plt.show()
#plt.savefig("xval_6panel.png")
