import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
# rc('text.latex', preamble = ','.join('''\usepackage{txfonts}'''.split()))
plt.rc('font', family='serif')
import numpy as np


def round_sig(x, sig=2):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)


names = ['\mbox{T}_{\mbox{eff}}', '\mbox{log g}', 
'\mbox{[Fe/H]}', 
'\mbox{[C/M]}', '\mbox{[N/M]}',
r'[\alpha/\mbox{M}]',
'\mbox{A}_{\mbox{k}}']
units = ['K', 'dex', 'dex', 'dex', 'dex','dex', 'mag']
mins = [3700, 1.0, -2.0, -0.3, -0.3,-0.11,-0.11]
maxs = [5500, 4.1, 0.6, 0.38, 0.5, 0.4, 0.4]

print("Loading data")
direc = "/Users/annaho/Data/LAMOST/Mass_And_Age"
data_direc = direc + "/with_col_mask/xval_with_cuts"
ref_ids = np.load("%s/ref_id.npz" %data_direc)['arr_0']
snr = np.load(
    "%s/ref_snr.npz" %data_direc)['arr_0']
apogee = np.load("%s/ref_label.npz" %data_direc)['arr_0']
cannon = np.load(
    "%s/xval_cannon_label_vals.npz" %data_direc)['arr_0']

print("Loading excised data")
data_direc = direc + "/with_col_mask/excised_obj"
add_ids = np.load("%s/excised_ids.npz" %data_direc)['arr_0']
add_snr = np.load(
    "%s/excised_snr.npz" %data_direc)['arr_0']
add_apogee = np.load("%s/excised_label.npz" %data_direc)['arr_0']
add_cannon = np.load(
    "%s/excised_all_cannon_labels.npz" %data_direc)['arr_0']

ref_ids = np.hstack((ref_ids, add_ids))
snr = np.hstack((snr, add_snr))
apogee = np.vstack((apogee, add_apogee))
cannon = np.vstack((cannon, add_cannon))

fig = plt.figure(figsize=(9,11))
gs = gridspec.GridSpec(4,2, wspace=0.3, hspace=0.5)
props = dict(boxstyle='round', facecolor='white', alpha=0.3)
props2 = dict(boxstyle='round', facecolor='white', alpha=0.3)

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
    #print(np.mean(cannon[:,i]-apogee[:,i]))
    choose = snr > 100
    diff = (cannon[:,i] - apogee[:,i])[choose]
    bias_raw = np.mean(diff)
    scatter_raw = np.std(diff)
    if np.abs(bias_raw) < 1:
        bias = round_sig(bias_raw, sig=2)
    else:
        bias = int(bias_raw)
    if np.abs(scatter_raw) < 1:
        scatter = round_sig(scatter_raw, sig=2)
    else:
        scatter = int(scatter_raw)
    textstr1 = "Bias: %s\nScatter: %s" %(bias, scatter)
    if i < 6:
        ax.hist2d(
                apogee[:,i], cannon[:,i], range=[[low,high],[low,high]], 
                bins=50, norm=LogNorm(), cmap="gray_r")
        ax.text(
                0.05, 0.95, textstr1, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=props)
    elif i ==6:
        ax.hist2d(
                apogee[:,i], cannon[:,i], range=[[low,high],[low,high]], 
                #bins=50, norm=LogNorm(), cmap="Purples", alpha=1.0)
                bins=50, norm=LogNorm(), cmap="gray_r", alpha=1.0)
        ax.text(
                0.05, 0.95, textstr1, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=props2)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    #ax.set_xlabel(r"$%s$" %name + " (%s) from Prev. X-Val" %unit)
    ax.set_xlabel(r"$%s$" %name + " (%s) from APOGEE" %unit)
    ax.set_ylabel(r"$%s$" %(name) + " (%s) from Cannon" %unit)

#plt.show()
plt.savefig("xval_5panel.png")
