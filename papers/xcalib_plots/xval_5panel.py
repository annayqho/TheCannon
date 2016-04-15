import matplotlib.pyplot as plt
from math import log10, floor
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
rc('text.latex', preamble = ','.join('''
\usepackage{txfonts}
'''.split()))
plt.rc('font', family='serif')
import numpy as np

def round_sig(x, sig=2):
    if x < 0:
        return -round(-x, sig-int(floor(log10(-x)))-1)
    return round(x, sig-int(floor(log10(x)))-1)

names = ['\mbox{T}_{\mbox{eff}}', '\mbox{log g}', '\mbox{[Fe/H]}', r'[\alphaup/\mbox{M}]', 
'\mbox{A}_{\mbox{k}}']
units = ['K', 'dex', 'dex', 'dex', 'mag']
mins = [3700, 0.5, -2.4, -0.11, -0.1]
maxs = [5500, 4.1, 0.6, 0.38, 0.5]

direc_orig = "../run_2_train_on_good"
direc = "../run_9_more_metal_poor"
direc = "../run_9b_reddening"
all_cannon = np.load("%s/all_cannon_labels.npz" %direc)['arr_0']
all_ids = np.load("%s/all_ids.npz" %direc_orig)['arr_0']
all_apogee = np.load("%s/all_label.npz" %direc_orig)['arr_0']
good_id = np.load("%s/tr_id.npz" %direc)['arr_0']
snr = np.load("%s/tr_snr.npz" %direc)['arr_0']

choose = np.array([np.where(all_ids==val)[0][0] for val in good_id])
#apogee = all_apogee[choose]
apogee = np.load("%s/tr_label.npz" %direc)['arr_0']
cannon = all_cannon

fig = plt.figure(figsize=(8,9))
gs = gridspec.GridSpec(3,2, wspace=0.3, hspace=0.3)
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
    choose = snr > 50
    diff = cannon[:,i][choose] - apogee[:,i][choose]
    bias = round_sig(np.mean(diff), sig=3)
    scatter = round_sig(np.std(diff), sig=3)
    textstr1 = "Bias: %s\nScatter: %s" %(bias, scatter)
    if i < 4:
        ax.hist2d(apogee[:,i], cannon[:,i], range=[[low,high],[low,high]], bins=50, norm=LogNorm(), cmap="gray_r")
        ax.text(0.05, 0.95, textstr1, 
                transform=ax.transAxes,fontsize=14, verticalalignment='top', bbox=props)
    elif i ==4:
        ax.hist2d(apogee[:,i], cannon[:,i], range=[[low,high],[low,high]], bins=50, norm=LogNorm(), cmap="Purples", alpha=1.0)
        ax.text(0.05, 0.95, textstr1, 
                transform=ax.transAxes,fontsize=14, verticalalignment='top', bbox=props2)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r"$%s$" %name + " (%s) from APOGEE" %unit)
    ax.set_ylabel(r"$%s$" %(name) + " (%s) from Cannon" %unit)

#plt.show()
plt.savefig("xval_5panel.png")
