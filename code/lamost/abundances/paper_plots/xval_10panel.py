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


#direc = "/home/annaho/TheCannon/data/LAMOST/abundances"
direc = "/Users/annaho/Data/LAMOST/Abundances"
label_names_all_h = np.load(direc + "/label_names.npz")['arr_0']
label_names_all_h = np.array([val.decode('utf-8') for val in label_names_all_h])
label_names_all_fe = np.array(["%s/Fe" %val for val in label_names_all_h[3:]])
label_names_all = np.hstack((label_names_all_h[0:3], label_names_all_fe))
label_names_all[6] = "Fe/H"

units_all = ['K']
for label in label_names_all[1:]:
    units_all.append('dex')

start = 7
end = len(label_names_all)
#start = 7
#end = 
names = label_names_all[start:end]
units = units_all[start:end]

all_cannon_raw = np.load(direc + "/xval_cannon_label_vals.npz")['arr_0']
all_ids = np.load(direc + "/ref_id.npz")['arr_0']
all_apogee_raw = np.load(direc + "/ref_label.npz")['arr_0']

cannon_feh = all_cannon_raw[:,6]
apogee_feh = all_apogee_raw[:,6]

all_cannon = np.zeros(all_cannon_raw.shape)
all_apogee = np.zeros(all_apogee_raw.shape)
all_cannon[:,0:3] = all_cannon_raw[:,0:3]
all_apogee[:,0:3] = all_apogee_raw[:,0:3]
all_cannon[:,3:] = all_cannon_raw[:,3:] - cannon_feh[:,None]
all_apogee[:,3:] = all_apogee_raw[:,3:] - apogee_feh[:,None]
all_apogee[:,6] = apogee_feh
all_cannon[:,6] = cannon_feh

snr = np.load(direc + "/ref_snr.npz")['arr_0']

apogee = all_apogee[:,start:end] 
cannon = all_cannon[:,start:end] 


#lows = [3800, 0, -1.7,-0.08, -1.0, -1.5, -0.8, -1.3, -1.5, -1.0,
#        -1.0, -1.5, -1.5, -1.2, -0.7, -0.7, -0.7, -1.1, -2.4]
lows = [3800, 0, -0.4, -0.2, -0.3, -0.3, -1.0, 
        -0.2, -0.2, -0.2, -0.2, -0.2, 0, -0.3]
lows = lows[start:end]
#highs = [5500, 5.1, 0.55, 0.41, 0.47, 0.86, 0.64, 0.56, 0.45, 0.44,
#        0.56, 0.58, 0.70, 0.44, 0.52, 0.64, 0.6, 0.62, 0.8]
highs = [5500, 5.1, 0.3, 0.5, 0.2, 0.2, 0.45, 
        0.3, 0.3, 0.2, 0.40, 0.4, 0.4, 0.3]
highs = highs[start:end]

def plot_tile(i,j):
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

    ax = plt.subplot(gs[j])
    #low = np.min([np.min(y_apogee), np.min(y_cannon)])
    #high = np.max([np.max(y_apogee), np.max(y_cannon)])
    low = lows[i]
    high = highs[i]
    print(low, high)
    ax.hist2d(
            y_apogee, y_cannon, norm=LogNorm(), 
            range=[[low,high],[low,high]], bins=60, cmap="gray_r")
    ax.plot([low,high], [low,high], c='k',label="x=y")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("APOGEE Value", fontsize=16)
    ax.set_ylabel(r"$%s \mathrm{(%s)}$" %(name,unit), fontsize=16)
    text = "Scatter: %s" %scatter
    ax.text(0.05, 0.90, text, 
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, fontsize=14)
    text = "Bias: %s" %bias
    ax.text(0.05, 0.80, text, 
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, fontsize=14)

fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(4,2, wspace=0.3, hspace=0.3)
plot_tile(0,0)
plot_tile(1,1)
plot_tile(2,2)
plot_tile(3,3)
plot_tile(4,4)
plot_tile(5,5)
plot_tile(6,6)
#plot_tile(7,7)

#plt.show()
plt.savefig("xval_mg_to_ti.png")
#plt.savefig("xval_teff_to_c.png")
