import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# make sample spectra
plot(dataset.wl, dataset.tr_flux[2,:], alpha=0.7, c='k')
title(r"Typical High-S/N LAMOST Spectrum", fontsize=27)
xlim(3500, 9500)
tick_params(axis='x', labelsize=27)
tick_params(axis='y', labelsize=27)
xlabel("Wavelength ($\AA$)", fontsize=27)
ylabel("Flux", fontsize=27)
savefig("typical_spec_snr186.png")

ID = "spec-55938-B5593806_sp04-159.fits"
# now find it in APOGEE...
ID = "aspcapStar-r5-v603-2M12252154+2732475.fits" 
import pyfits
fits_file = ID
file_in = pyfits.open(fits_file)
flux = np.array(file_in[1].data)
npixels = len(flux)
start_wl = file_in[1].header['CRVAL1']
diff_wl = file_in[1].header['CDELT1']
val = diff_wl * (npixels) + start_wl
wl_full_log = np.arange(start_wl,val, diff_wl)
wl_full = [10 ** aval for aval in wl_full_log]
wl = np.array(wl_full)
bad = flux == 0
wl = np.ma.array(wl, mask=bad)
flux = np.ma.array(flux, mask=bad)
plot(wl, flux, alpha=0.7, c='k')
xlim(15100, 17000)
ylim(0.6, 1.15)
title(r"Typical High-S/N APOGEE Spectrum", fontsize=27)
tight_layout()
savefig("typical_spec_snr186_apogee.png")




label_file = 'reference_labels.csv'
(test_ID, test_SNR) = pickle.load(open("test_ID_SNR.p", "r"))

# for each test ID, find its index in label_file IDs
ids = np.loadtxt(label_file, usecols=(0,), dtype=str, delimiter=',')
inds = [np.where(ids==test_ID_val) for test_ID_val in test_ID]

names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]']
lims = [[3900,6000], [0,5], [-2, 1], [-0.1,0.4]] 
#id,teff,logg,feh,alpha,snr
teff = np.loadtxt(label_file, usecols=(2,), dtype=float, delimiter=',')
logg = np.loadtxt(label_file, usecols=(3,), dtype=float, delimiter=',')
feh = np.loadtxt(label_file, usecols=(4,), dtype=float, delimiter=',')
alpha = np.loadtxt(label_file, usecols=(5,), dtype=float, delimiter=',')

apogee_label_vals = np.vstack(
        (teff[inds].flatten(), logg[inds].flatten(), feh[inds].flatten(), alpha[inds].flatten())).T
test_labels = pickle.load(open("test_labels.p", "r")) 

for i in range(0, len(names)):
    name = names[i]
    cannon = np.array(test_labels[:,i])
    orig = np.array(apogee_label_vals[:,i], dtype=float)
    snr = test_SNR
    #bad = orig < -8000
    #good = snr > 50
    #orig = np.ma.array(orig, mask=bad)
    #cannon = np.ma.array(cannon, mask=bad)
    #snr = np.ma.array(snr, mask=bad)
    #orig = orig[good]
    #cannon = cannon[good]
    #snr = snr[good]

    scatter = np.round(np.std(orig-cannon),3)
    scatter = int(scatter)
    bias  = np.round(np.mean(orig-cannon),4)
    bias = np.round(bias, 3)
    low = np.minimum(min(orig), min(cannon))
    high = np.maximum(max(orig), max(cannon))

    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(1,2,width_ratios=[2,1], wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
    low = lims[i][0]
    high = lims[i][1]
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    c = np.zeros(len(snr))
    take = snr < 100
    ax1.scatter(orig[take], cannon[take], marker='x', c='0.10', alpha=0.3, label="snr < 100")
    take = snr > 100
    ax1.scatter(orig[take], cannon[take], marker='x', c='k', label="snr > 100", alpha=0.7)
    ax1.legend(fontsize=14, loc='lower right')
    textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
            fontsize=14, verticalalignment='top')
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlabel("APOGEE $%s$" %name, fontsize=14)
    ax1.set_ylabel("Cannon-LAMOST $%s$" %name, fontsize=14)
    ax1.set_title("Cannon-LAMOST Output vs. APOGEE $%s$ " %name, fontsize=14)
    diff = cannon - orig
    npoints = len(diff)
    mu = np.mean(diff)
    sig = np.std(diff)
    ax2.hist(diff, range=[-3*sig,3*sig], color='k', bins=np.sqrt(npoints),
            orientation='horizontal', alpha=0.3, histtype='stepfilled')
    textstr = r"$\sigma=%s$" %(np.round(sig,2))
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
            fontsize=14, verticalalignment='top')
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_xlabel("Count", fontsize=14)
    ax2.set_ylabel("Difference", fontsize=14)
    ax2.axhline(y=0, c='k', lw=3, label='Difference=0')
    ax2.set_title("Cannon-LAMOST Output Minus \n APOGEE Labels for $%s$" %name,
            fontsize=14)
    ax2.legend(fontsize=14, loc='lower center')
    plt.savefig('1to1_%s.png'%i)


