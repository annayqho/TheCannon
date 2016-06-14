import numpy as np
from matplotlib.ticker import LogFormatter
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

files = glob.glob("output/*all_cannon_labels.npz")
chisq = glob.glob("output/*cannon_label_chisq.npz")

teff_all = []
logg_all = []
feh_all = []
alpha_all = []
logm_all = []
ak_all = []
chisq_all = []

for i,f in enumerate(files):
    date = f[7:].split("_")[0]
    label = np.load(f)['arr_0']
    nobj = label.shape[0]
    searchfor = "../xcalib_4labels/output/*%s*SNR*.npz" %date
    chisq_val = np.load(chisq[i])['arr_0']
    chisq_all.extend(chisq_val)
    teff_all.extend(label[:,0])
    logg_all.extend(label[:,1])
    feh_all.extend(label[:,2])
    alpha_all.extend(label[:,3])
    logm_all.extend(label[:,4])
    ak_all.extend(label[:,5])

teff_all = np.array(teff_all)
logg_all = np.array(logg_all)
feh_all = np.array(feh_all)
alpha_all = np.array(alpha_all)
logm_all = np.array(logm_all)
ak_all = np.array(ak_all)

np.savez("labels_all.npz", np.vstack((teff_all, logg_all, feh_all, alpha_all, logm_all, ak_all)))

fig,axarr = plt.subplots(1,3, figsize=(12,5), sharex=True, sharey=True)
snr_min = [30,60,100]

for i in range(0, len(snr_min)):
    ax = axarr[i]
    snr_cut = snr > snr_min[i]
    # to remain within the reference space:
    teff_cut = np.logical_and(teff_all>4000, teff_all<5200)
    logg_cut = np.logical_and(logg_all>1.3, logg_all<3.3)
    feh_cut = np.logical_and(feh_all>-0.8, feh_all<0.4)

    choose_1 = np.logical_and(snr_cut, logg_cut)
    choose_2 = np.logical_and(feh_cut, teff_cut)
    choose = np.logical_and(choose_1, choose_2)

    print(len(choose))
    num = sum(choose)
    #plt.scatter(feh_all, alpha_all, c=age, edgecolor='none', s=1, norm=LogNorm(), vmin=1, vmax=13)
    xmin = -1.0
    xmax = 0.7
    ymin = -0.15
    ymax = 0.5
    im = ax.hexbin(feh_all[choose], alpha_all[choose], C=10**(logm_all[choose]), gridsize=60, extent=(xmin,xmax,ymin,ymax), reduce_C_function = np.median, vmin=1, vmax=12, norm=LogNorm())
    ax.text(0.05, 0.90, r"SNR \textgreater %s (%s objects)" %(snr_min[i], num), 
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    if i == 1:
        ax.set_xlabel(r"$[Fe/H]$" + r" (dex) from Cannon/LAMOST", fontsize=16)
    elif i == 0:
        ax.set_ylabel(r"$[\alpha/Fe]$" + "(dex) \nfrom Cannon/LAMOST", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    
formatter = LogFormatter(10,labelOnlyBase=False)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92,0.1,0.02,0.8])
fig.colorbar(im, cax=cbar_ax, label="Age (Gyr)", format=formatter, ticks=[1,2,3,5,7,9,12])
plt.savefig("feh_alpha_cmass.png" %snr_min)
#plt.show()
