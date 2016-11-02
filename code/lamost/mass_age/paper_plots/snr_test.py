import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import pyfits
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

names = ['\mbox{T}_{\mbox{eff}}', '\mbox{log g}', '\mbox{[M/H]}',
'\mbox{[C/M]}', '\mbox{[N/M]}', r'[\alpha/\mbox{M}]']
units = ['K', 'dex', 'dex', 'dex', 'dex', 'dex']

direc = "/Users/annaho/Data/LAMOST/Mass_And_Age"
data_direc = direc + "/with_col_mask/xval_with_cuts"
ref_ids = np.load("%s/ref_id.npz" %data_direc)['arr_0']
snr = np.load("%s/ref_snr.npz" %data_direc)['arr_0']
apogee = np.load("%s/ref_label.npz" %data_direc)['arr_0']
cannon = np.load(
        "%s/xval_cannon_label_vals.npz" %data_direc)['arr_0']

data_direc = direc + "/with_col_mask/excised_obj"
add_ids = np.load("%s/excised_ids.npz" %data_direc)['arr_0']
add_snr = np.load(
            "%s/excised_snr.npz" %data_direc)['arr_0']
add_apogee = np.load("%s/excised_label.npz" %data_direc)['arr_0']
add_cannon = np.load(
            "%s/excised_all_cannon_labels.npz" %data_direc)['arr_0']

ref_ids = np.hstack((ref_ids, add_ids))
snr = np.hstack((snr, add_snr))
apogee = np.vstack((apogee, add_apogee))
cannon = np.vstack((cannon, add_cannon))

hdulist = pyfits.open(direc + "/catalog_paper.fits")
tbdata = hdulist[1].data
hdulist.close()
snrg = tbdata.field("snr")
is_ref = tbdata.field("is_ref_obj")
choose = is_ref == 1.
lamost_id = tbdata.field("lamost_id_1")[choose]
inds = np.array([np.where(lamost_id==val)[0][0] for val in ref_ids])
lamost_teff = tbdata.field("teff")[choose][inds]
lamost_logg = tbdata.field("logg")[choose][inds]
lamost_feh = tbdata.field("feh")[choose][inds]
lamost = np.vstack((lamost_teff, lamost_logg, lamost_feh)).T

print("Loading excised data")

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3,2, wspace=0.3, hspace=0.3)

lows = [40, 0.09, 0.04, 0.055, 0.08, 0.025]
highs = [150, 0.40, 0.17, 0.11, 0.16, 0.06]
offsets = np.array([40, 0.09, 0.04, 0.05, 0.072, 0.025])

snr = snrg[choose]
snr_label = r"S/N"

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
