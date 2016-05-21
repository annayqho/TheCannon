import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]']
units = ['K', 'dex', 'dex', 'dex']

direc = "../run_9_more_metal_poor/"
all_cannon = np.load("%s/all_cannon_labels.npz" %direc)['arr_0']
all_ids = np.load("../run_2_train_on_good/all_ids.npz")['arr_0']
all_apogee = np.load("../run_2_train_on_good/all_label.npz")['arr_0']
good_id = np.load("%s/tr_id.npz" %direc)['arr_0']
snr = np.load("%s/tr_snr.npz" %direc)['arr_0']

IDs_lamost = np.loadtxt(
        "../../examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt",
        usecols=(0,), dtype=(str))
labels_all_lamost = np.loadtxt(
        "../../examples/test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt",
        usecols=(3,4,5), dtype=(float))
inds = np.array([np.where(IDs_lamost==a)[0][0] for a in good_id])
lamost = labels_all_lamost[inds,:]

choose = np.array([np.where(all_ids==val)[0][0] for val in good_id])
apogee = all_apogee[choose]
cannon = all_cannon

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,2, wspace=0.3, hspace=0.3)

b = 70
K = [b*80, 0.2*b, 0.10*b, 0.03*b]
lows = [60, 0.10, 0.06, 0.035]
highs = [180, 0.45, 0.21, 0.07]

obj = []

for i in range(0, len(names)):
    name = names[i]
    unit = units[i]
    #low = mins[i]
    #high = maxs[i]
    
    snr_bins = np.array([10,30,50,70,90,110])
    y_cannon = np.zeros(len(snr_bins))
    y_lamost = np.zeros(len(snr_bins))
    yerr_cannon = np.zeros(len(snr_bins))
    yerr_lamost = np.zeros(len(snr_bins))
    for ii,center in enumerate(snr_bins):
        choose = np.abs(snr-center)<10
        diff_cannon = cannon[:,i][choose]-apogee[:,i][choose]
        if i < len(names)-1:
            diff_lamost = lamost[:,i][choose]-apogee[:,i][choose]
        else:
            diff_lamost = np.zeros(len(diff_cannon))
        # bootstrap 100 times
        nbs = 100
        nobj = len(diff_cannon)
        samples = np.random.randint(0,nobj,(nbs,nobj))
        stdev = np.std(diff_cannon[samples], axis=1)
        y_cannon[ii] = np.mean(stdev)
        yerr_cannon[ii] = np.std(stdev)
        stdev = np.std(diff_lamost[samples], axis=1)
        y_lamost[ii] = np.mean(stdev)
        yerr_lamost[ii] = np.std(stdev)

    ax = plt.subplot(gs[i])
    ax.scatter(snr_bins, y_cannon)
    obj.append(ax.errorbar(snr_bins, y_cannon, yerr=yerr_cannon, fmt='.', c='darkorchid', label="Cannon from LAMOST spectra"))
    ax.scatter(snr_bins, y_lamost)
    obj.append(ax.errorbar(snr_bins, y_lamost, yerr=yerr_lamost, fmt='.', c='darkorange', label="Cannon from LAMOST spectra"))
    # a 1/r^2 line
    xfit = np.linspace(min(snr_bins), max(snr_bins))
    yfit = K[i] / xfit
    obj.append(ax.plot(xfit,yfit,c='k', label="1/SNR")[0])
    ax.set_xlim(0, 120)
    ax.set_ylim(lows[i], highs[i])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("SNR", fontsize=16)
    ax.set_ylabel(r"$\sigma %s \mathrm{(%s)}$" %(name,unit), fontsize=16)

fig.legend((obj[0],obj[1],obj[2]), ("Cannon", "LAMOST", "1/SNR"), fontsize=16)

#plt.show()
plt.savefig("snr_test_4panel.png")
