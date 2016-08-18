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


direc = "/home/annaho/TheCannon/data/LAMOST/abundances"
label_names_all = np.load(direc + "/label_names.npz")['arr_0']

units_all = ['K']
for label in label_names_all[1:]:
    units_all.append('dex')

#start = 8
#end = len(label_names_all)
start = 3
end = 8
names = label_names_all[start:end]
names = np.array(["%s/Fe" %val for val in names])
units = units_all[start:end]

all_cannon_raw = np.load(direc + "/xval_cannon_label_vals.npz")['arr_0']
all_ids = np.load(direc + "/ref_id.npz")['arr_0']
all_apogee_raw = np.load(direc + "/ref_label.npz")['arr_0']

cannon_feh = all_cannon_raw[:,6]
all_cannon = all_cannon_raw - cannon_feh[:,None]
apogee_feh = all_apogee_raw[:,6]
all_apogee = all_apogee_raw - apogee_feh[:,None]
all_apogee[:,6] = all_apogee_raw[:,6]
all_cannon[:,6] = all_cannon_raw[:,6]

#good = np.min(all_cannon, axis=1) > -500
#all_cannon = all_cannon[good]
#good_id = all_ids[good]
#all_apogee = all_apogee[good]

snr = np.load(direc + "/ref_snr.npz")['arr_0']

apogee = all_apogee[:,start:end] 
cannon = all_cannon[:,start:end] 

fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(4,2, wspace=0.3, hspace=0.3)

#lows = [3800, 0, -1.7,-0.08, -1.0, -1.5, -0.8, -1.3, -1.5, -1.0,
#        -1.0, -1.5, -1.5, -1.2, -0.7, -0.7, -0.7, -1.1, -2.4]
#lows = [3800, 0, -0.4, -0.5, -0.2, -0.3, -1.0, -0.4, -0.2, -0.3, 
#        -0.3, -0.15, -0.1, -0.0, -0.0, -0.2, -0.4]
#lows = lows[start:end]
#highs = [5500, 5.1, 0.55, 0.41, 0.47, 0.86, 0.64, 0.56, 0.45, 0.44,
#        0.56, 0.58, 0.70, 0.44, 0.52, 0.64, 0.6, 0.62, 0.8]
#highs = [5500, 5.1, 0.3, 0.6, 0.5, 0.3, 0.45, 0.2, 0.3, 0.3, 
#        0.30, 0.1, 0.4, 0.5, 0.5, 0.2, 0.4]
#highs = highs[start:end]

obj = []

for i in range(0, len(names)):
    name = names[i]
    unit = units[i]
    #low = mins[i]
    #high = maxs[i]
    
    choose = snr > 100
    diff_cannon = cannon[:,i][choose]-apogee[:,i][choose]
    scatter = round_sig(np.std(diff_cannon),3)
    bias = round_sig(np.mean(diff_cannon),3)
    y_cannon = cannon[:,i][choose]
    y_apogee = apogee[:,i][choose]

    ax = plt.subplot(gs[i])
    low = np.min([np.min(y_apogee), np.min(y_cannon)])
    high = np.max([np.max(y_apogee), np.max(y_cannon)])
    #low = lows[i]
    #high = highs[i]
    print(low, high)
    ax.hist2d(
            y_apogee, y_cannon, norm=LogNorm(), 
            range=[[low,high],[low,high]], bins=60, cmap="gray_r")
    ax.plot([low,high], [low,high], c='k',label="x=y")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if i > 7:
        ax.set_xlabel("APOGEE Value", fontsize=16)
    ax.set_ylabel(r"$%s \mathrm{(%s)}$" %(name,unit), fontsize=16)
    text = "Scatter: %s" %scatter
    ax.text(0.05, 0.90, text, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    text = "Bias: %s" %bias
    ax.text(0.05, 0.70, text, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)


#plt.show()
plt.savefig("xval_last.png")
#plt.savefig("xval_first.png")
