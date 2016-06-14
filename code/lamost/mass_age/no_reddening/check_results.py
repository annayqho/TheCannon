import numpy as np
from matplotlib.ticker import LogFormatter
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm
from matplotlib import rc
from mass_age_functions import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

files = glob.glob("output/*all_cannon_labels.npz")
chisq = glob.glob("output/*cannon_label_chisq.npz")
#snr_files = glob.glob("output/*snr.npz")

teff_all = []
logg_all = []
feh_all = []
alpha_all = []
cm_all = []
nm_all = []
chisq_all = []
snr = []

for i,f in enumerate(files):
    date = f[7:].split("_")[0]
    snr_file = glob.glob("output/*%s*snr*.npz" %date)[0]
    snr_vals = np.load(snr_file)['arr_0']
    snr.extend(snr_vals)
    label = np.load(f)['arr_0']
    chisq_val = np.load(chisq[i])['arr_0']
    chisq_all.extend(chisq_val)
    teff_all.extend(label[:,0])
    logg_all.extend(label[:,1])
    feh_all.extend(label[:,2])
    alpha_all.extend(label[:,3])
    cm_all.extend(label[:,4])
    nm_all.extend(label[:,5])

teff_all = np.array(teff_all)
logg_all = np.array(logg_all)
feh_all = np.array(feh_all)
alpha_all = np.array(alpha_all)
cm_all = np.array(cm_all)
nm_all = np.array(nm_all)
snr = np.array(snr)

age = 10**calc_logAge(feh_all, cm_all, nm_all, teff_all, logg_all)

fig,axarr = plt.subplots(1,3, figsize=(12,5), sharex=True, sharey=True)
snr_min = [30,60,100]

for i in range(0, len(snr_min)):
    ax = axarr[i]
    snr_cut = snr > snr_min[i]
    logg_cut = np.logical_and(logg_all>1.8, logg_all<3.3) # to include only post dredge up giants
    # to remain within the reference space of the APOKASC sample:
    feh_cut = feh_all>-0.8
    teff_cut = np.logical_and(teff_all>4000, teff_all<5000)
    cm_cut = np.logical_and(cm_all>-0.25, cm_all<0.15)
    nm_cut = np.logical_and(nm_all>-0.1, nm_all<0.45)
    cn_cut = np.logical_and(cm_all-nm_all>-0.6, cm_all-nm_all<0.2)
    cn_sum = calc_sum(feh_all,cm_all,nm_all)
    cn_sum_cut = np.logical_and(cn_sum>-0.1, cn_sum<0.15)

    choose_1 = np.logical_and(snr_cut, logg_cut)
    #choose_2 = feh_cut
    choose_2 = np.logical_and(feh_cut, teff_cut)
    choose_3 = np.logical_and(cm_cut, nm_cut)
    choose_4 = np.logical_and(cn_cut, cn_sum_cut)
    choose_12 = np.logical_and(choose_1, choose_2)
    choose_34 = np.logical_and(choose_3, choose_4)

    choose = np.logical_and(choose_12, choose_34)
    ##choose = snr_cut
    #choose = np.logical_and(snr_cut, logg_cut)
    print(len(choose))
    num = sum(choose)
    #plt.scatter(feh_all, alpha_all, c=age, edgecolor='none', s=1, norm=LogNorm(), vmin=1, vmax=13)
    xmin = -1.0
    xmax = 0.7
    ymin = -0.15
    ymax = 0.5
    im = ax.hexbin(feh_all[choose], alpha_all[choose], C=age[choose], gridsize=60, extent=(xmin,xmax,ymin,ymax), reduce_C_function = np.median, vmin=1, vmax=12, norm=LogNorm())
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
plt.savefig("feh_alpha_logg_all.png" %snr_min)
#plt.show()
