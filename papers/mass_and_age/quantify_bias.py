import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]', 'AKWISE']
units = ['K', 'dex', 'dex', 'dex', 'dex']
mins = np.array([3900, 0.7, -1.6, 0.0, -0.1])
maxs = np.array([5300, 3.9, 0.3, 0.3, 0.4])
nbins = 10 
chunks = (maxs-mins)/nbins
bins = mins[:,None]+np.arange(nbins)[None,:]*chunks[:,None] # (nlabel, nbin)

direc_orig = "../run_2_train_on_good"
direc = "../run_9_more_metal_poor"
direc = "../run_9b_reddening"
all_cannon = np.load("%s/all_cannon_labels.npz" %direc)['arr_0']
all_ids = np.load("%s/all_ids.npz" %direc_orig)['arr_0']
#all_apogee = np.load("%s/all_label.npz" %direc_orig)['arr_0']
good_id = np.load("%s/tr_id.npz" %direc)['arr_0']
snr = np.load("%s/tr_snr.npz" %direc)['arr_0']

#choose = np.array([np.where(all_ids==val)[0][0] for val in good_id])
#apogee = all_apogee[choose]
apogee = np.load("%s/tr_label.npz" %direc)['arr_0']
cannon = all_cannon

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3,2, wspace=0.3, hspace=0.3)

for i in range(0, len(names)):
    name = names[i]
    unit = units[i]
    low = mins[i]
    high = maxs[i]
    val = cannon[:,i]-apogee[:,i]

    y = np.zeros(nbins)
    yerr = np.zeros(y.shape)
    x = np.zeros(y.shape)
    for ii,start in enumerate(bins[i]):
        end = start + chunks[i]
        x[ii] = (end+start)/2
        choose = np.logical_and(apogee[:,i]<end, apogee[:,i]>start)
        print(sum(choose))
        diff = val[choose]
        # bootstrap 100 times
        nbs = 100
        nobj = len(diff)
        samples = np.random.randint(0,nobj,(nbs,nobj))
        bias = np.median(diff[samples], axis=1) 
        y[ii] = np.mean(bias) # bias
        yerr[ii] = np.std(bias)

    ax = plt.subplot(gs[i])
    if i == 3:
        ax.axvline(x=0.09, c='k')
        ax.set_ylim(-0.02,0.01)
    ax.scatter(x, y)
    ax.errorbar(x, y, yerr=yerr, fmt='.', c='darkorchid', label="Bias between Cannon and APOGEE Value")
    #ax.set_xlim(low, high)
    #ax.set_ylim(low, high)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel("Med(Cannon-APOGEE), " + r"$%s$" %name + " (%s)" %unit)
    ax.set_xlabel("APOGEE " + r"$%s$" %(name) + " (%s)" %unit)
    #ax.set_xlabel(r"$%s$" %name + " (%s) from APOGEE" %unit)
    #ax.set_ylabel(r"$%s$" %(name) + " (%s) from Cannon/LAMOST" %unit)

plt.show()
#plt.savefig("bias_4panel.png")
