import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import pyfits
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

names = ['\mbox{T}_{\mbox{eff}}', '\mbox{log g}', '\mbox{[Fe/H]}',
'\mbox{[C/M]}', '\mbox{[N/M]}', r'[\alpha/\mbox{M}]']
units = ['K', 'dex', 'dex', 'dex', 'dex', 'dex']

hdulist = pyfits.open(
    "/Users/annaho/Data/Mass_And_Age/lamost_catalog_mass_age.fits")
tbdata = hdulist[1].data
hdulist.close()
snrg = tbdata.field("snrg")
is_ref = tbdata.field("is_ref_obj")
choose = is_ref == 1.
lamost_id = tbdata.field("LAMOST_ID_1")[choose]
lamost_teff = tbdata.field("teff_1")[choose]
lamost_logg = tbdata.field("logg_1")[choose]
lamost_feh = tbdata.field("feh")[choose]
lamost = np.vstack((lamost_teff, lamost_logg, lamost_feh)).T
cannon_teff = tbdata.field("cannon_teff_1")[choose]
cannon_logg = tbdata.field("cannon_logg_1")[choose]
cannon_feh = tbdata.field("cannon_mh")[choose]
cannon_cm = tbdata.field("cannon_cm")[choose]
cannon_nm = tbdata.field("cannon_nm")[choose]
cannon_afe = tbdata.field("cannon_afe")[choose]
apogee_teff = tbdata.field("apogee_teff")[choose]
apogee_logg = tbdata.field("apogee_logg")[choose]
apogee_mh = tbdata.field("apogee_mh")[choose]
apogee_cm = tbdata.field("apogee_cm")[choose]
apogee_nm = tbdata.field("apogee_nm")[choose]
apogee_afe = tbdata.field("apogee_afe")[choose]

cannon = np.vstack((
    cannon_teff, cannon_logg, cannon_feh, cannon_cm, cannon_nm, cannon_afe)).T
apogee = np.vstack((
    apogee_teff, apogee_logg, apogee_mh, apogee_cm, apogee_nm, apogee_afe)).T

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3,2, wspace=0.3, hspace=0.3)

lows = [45, 0.05, 0.03, 0.06, 0.08, 0.025]
highs = [145, 0.40, 0.17, 0.11, 0.15, 0.06]
offsets = np.array([50, 0.1, 0.05, 0.06, 0.08, 0.03])

snr = snrg[choose]
snr_label = r"$\sim$\,SNRg"

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
            diff_lamost = np.zeros(len(diff_cannon)) - 100
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
    obj.append(ax.errorbar(
        snr_bins, y_lamost, yerr=yerr_lamost, fmt='.', 
        c='darkorange', label="Cannon from LAMOST spectra"))
    ax.scatter(snr_bins, y_cannon)
    obj.append(ax.errorbar(
        snr_bins, y_cannon, yerr=yerr_cannon, fmt='.', 
        c='darkorchid', label="Cannon from LAMOST spectra"))
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

fig.legend(
        (obj[0],obj[1],obj[2]), 
        ("LAMOST", "Cannon", "1/(%s)" %snr_label), 
        fontsize=16)

#plt.show()
plt.savefig("snr_test.png")
