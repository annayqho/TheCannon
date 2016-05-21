import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import pyfits
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
rc('text.latex', preamble = ','.join('''
    \usepackage{txfonts}
    \usepackage{lmodern}
    '''.split()))
plt.rc('font', family='serif')
import numpy as np

#names = ['\mbox{T}_{\mbox{eff}}', '\mbox{log g}', '\mbox{[Fe/H]}', r'[\alphaup/\mbox{Fe}]', 
#'\mbox{A}_{\mbox{k}}']
names = ['\mbox{T}_{\mbox{eff}}', '\mbox{log g}', '\mbox{[Fe/H]}', r'[\alphaup/\mbox{M}]']
#units = ['K', 'dex', 'dex', 'dex', 'mag']
units = ['K', 'dex', 'dex', 'dex']

all_ids = np.load("../run_2_train_on_good/all_ids.npz")['arr_0']
all_apogee = np.load("../run_2_train_on_good/all_label.npz")['arr_0']

hdulist = pyfits.open("../make_lamost_catalog/lamost_catalog_training.fits")
tbdata = hdulist[1].data
hdulist.close()
snrg = tbdata.field("snrg")
snri = tbdata.field("snri")
lamost_id_full = tbdata.field("lamost_id")
lamost_id = np.array([val.strip() for val in lamost_id_full])
lamost_teff = tbdata.field("teff_1")
lamost_logg = tbdata.field("logg_1")
lamost_feh = tbdata.field("feh")
lamost = np.vstack((lamost_teff, lamost_logg, lamost_feh)).T
cannon_teff = tbdata.field("cannon_teff")
cannon_logg = tbdata.field("cannon_logg")
cannon_feh = tbdata.field("cannon_m_h")
cannon_afe = tbdata.field("cannon_alpha_m")

cannon = np.vstack((cannon_teff, cannon_logg, cannon_feh, cannon_afe)).T
inds = np.array([np.where(all_ids==val)[0][0] for val in lamost_id])
apogee = all_apogee[inds]


fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,2, wspace=0.3, hspace=0.3)

lowsg = [50, 0.10, 0.06, 0.025]
highsg = [145, 0.40, 0.17, 0.063]
lowsi = [70, 0.14, 0.07, 0.047]
highsi = [160, 0.4, 0.2, 0.065]
offsetsg = np.array([50, 0.1, 0.06, 0.03])
offsetsi = np.array([70, 0.14, 0.08, 0.045])

snr = snrg
snr_label = r"$\sim$\,SNRg"
lows = lowsg
highs = highsg
offsets = offsetsg

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
        if i < 3:
            print(i)
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
    ax.scatter(snr_bins, y_lamost)
    obj.append(ax.errorbar(snr_bins, y_lamost, yerr=yerr_lamost, fmt='.', c='darkorange', label="Cannon from LAMOST spectra"))
    ax.scatter(snr_bins, y_cannon)
    obj.append(ax.errorbar(snr_bins, y_cannon, yerr=yerr_cannon, fmt='.', c='darkorchid', label="Cannon from LAMOST spectra"))
    # a 1/r^2 line
    xfit = np.linspace(0.1, max(snr_bins)*2, 100)
    b = 30
    K = b * y_cannon[1] 
    yfit = (K / (xfit*3)) 
    #yfit = (K / (xfit)) 
    offset = offsets[i]
    obj.append(ax.plot(xfit,yfit+offset,c='k', label="1/(%s)" %snr_label)[0])
    #obj.append(ax.plot(xfit,yfit,c='k', label="1/%s" %snr_label)[0])
    ax.set_xlim(0, 120)
    ax.set_ylim(lows[i], highs[i])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("%s" %snr_label, fontsize=16)
    ax.set_ylabel(r"$\sigma %s \mathrm{(%s)}$" %(name,unit), fontsize=16)

fig.legend((obj[0],obj[1],obj[2]), ("LAMOST", "Cannon", "1/(%s)" %snr_label), fontsize=16)

#plt.show()
#plt.savefig("%s_test_4panel_stretched.png" %snr_label[8:])
plt.savefig("snr_test_4panel.png")
