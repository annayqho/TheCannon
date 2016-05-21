import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm
from matplotlib import rc
plt.rc('text', usetex=True)
rc('text.latex', preamble = ','.join('''
    \usepackage{txfonts}
    \usepackage{lmodern}
    '''.split()))
plt.rc('font', family='serif')


files = glob.glob("output/*all_cannon_labels.npz")
chisq = glob.glob("output/*cannon_label_chisq.npz")

feh_all = []
#teff_all = []
alpha_all = []
chisq_all = []

for i,f in enumerate(files):
    labels = np.load(f)['arr_0']
    chisq_val = np.load(chisq[i])['arr_0']
    chisq_all.extend(chisq_val)
    feh = labels[:,2]
    feh_all.extend(feh)
    alpha = labels[:,5]
    alpha_all.extend(alpha)
    #teff = labels[:,0]
    #teff_all.extend(teff)

tr_feh = np.load("tr_label.npz")['arr_0'][:,2]
tr_afe = np.load("tr_label.npz")['arr_0'][:,3]

feh_all = np.array(feh_all)
alpha_all = np.array(alpha_all)
#teff_all = np.array(teff_all)
print("%s objects so far" %len(feh_all))
#plt.hist2d(feh_all, alpha_all, norm=LogNorm(), cmap="gray_r", bins=50)
plt.hist2d(feh_all, alpha_all, norm=LogNorm(), cmap="gray_r", bins=60, range=[[-2.2,.9],[-.2,.6]])
#choose = teff_all < 4000
#plt.scatter(feh_all, alpha_all, c=teff_all, edgecolor='none', s=1, vmin=3500, vmax=5500)
#plt.scatter(tr_feh, tr_afe, edgecolor='none', c='red', s=1, label="training set")
plt.xlabel("[Fe/H] (dex)" + " from Cannon/LAMOST", fontsize=16)
plt.ylabel(r"$\mathrm{[\alphaup/M]}$" + " (dex) from Cannon/LAMOST", fontsize=16)
plt.ylim(-0.2,0.5)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
#plt.legend()
plt.savefig("feh_alpha_temp.png")
