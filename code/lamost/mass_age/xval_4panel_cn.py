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


names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]', '[C/M]', '[N/M]']
units = ['K', 'dex', 'dex', 'dex', 'dex', 'dex']
mins = [3700, 0.5, -2.4, -0.11, -0.3, -0.4]
maxs = [5500, 4.1, 0.6, 0.38, 0.2, 0.55]
names = names[4:6]
units = units[4:6]
mins = mins[4:6]
maxs = maxs[4:6]

cannon = np.load("run_10_c_n_xcalib_highSNR/all_cannon_labels.npz")['arr_0']
all_id = np.load("run_10_c_n_xcalib_highSNR/tr_id.npz")['arr_0']
apogee = np.load("run_10_c_n_xcalib_highSNR/tr_label.npz")['arr_0']
snr = np.load("run_10_c_n_xcalib_highSNR/tr_snr.npz")['arr_0']
cannon = cannon[:,4:6]
apogee = apogee[:,4:6]

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,2, wspace=0.3, hspace=0.3)

i_vals = [0,1]

for j,i in enumerate(i_vals):
    name = names[j]
    unit = units[j]
    low = mins[j]
    high = maxs[j]
    #low = np.minimum(min(apogee[:,i]), min(cannon[:,i]))
    #high = np.maximum(max(apogee[:,i]), max(cannon[:,i]))
    ax = plt.subplot(gs[i])
    ax.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
    #ax.legend(fontsize=14)
    choose = snr > 100
    ax.hist2d(apogee[:,j][choose], cannon[:,j][choose], range=[[low,high],[low,high]], bins=50, norm=LogNorm(), cmap="gray_r")
    text = r"SNR \textgreater 100"
    ax.text(0.05, 0.90, text, horizontalalignment='left', verticalalignment='top', 
            transform=ax.transAxes)
    bias = round_sig(np.mean(cannon[:,j][choose]-apogee[:,j][choose]), 3)
    scatter = round_sig(np.std(cannon[:,j][choose]-apogee[:,j][choose]), 3)
    text = "bias = %s, \nscatter = %s" %(bias, scatter)
    ax.text(0.05, 0.80, text, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r"$%s$" %name + " (%s) from APOGEE" %unit)
    ax.set_ylabel(r"$%s$" %(name) + " (%s) from Cannon/LAMOST" %unit)
    ax = plt.subplot(gs[i+2])
    ax.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
    #ax.legend(fontsize=14)
    choose = snr > 50
    ax.hist2d(apogee[:,j][choose], cannon[:,j][choose], range=[[low,high],[low,high]], bins=50, norm=LogNorm(), cmap="gray_r")
    text = r"SNR \textgreater 50"
    ax.text(0.05, 0.90, text, horizontalalignment='left', verticalalignment='top', 
            transform=ax.transAxes)
    bias = round_sig(np.mean(cannon[:,j][choose]-apogee[:,j][choose]), 3)
    scatter = round_sig(np.std(cannon[:,j][choose]-apogee[:,j][choose]), 3)
    text = "bias = %s, \nscatter = %s" %(bias, scatter)
    ax.text(0.05, 0.80, text, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r"$%s$" %name + " (%s) from APOGEE" %unit)
    ax.set_ylabel(r"$%s$" %(name) + " (%s) from Cannon/LAMOST" %unit)

#plt.show()
plt.savefig("xval_4panel.png")
