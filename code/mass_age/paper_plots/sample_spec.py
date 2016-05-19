# Plot a sample LAMOST, APOGEE, continuum normalized spectrum

import numpy as np
import matplotlib.pyplot as plt
import pyfits
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

direc = "/home/annaho/aida41040/annaho/TheCannon/examples/test_training_overlap"
tr_ID = np.load("%s/tr_IDs.npz" %direc)['arr_0']
lamost_wl = np.load("%s/tr_data_raw.npz" %direc)['arr_1']
tr_flux = np.load("%s/tr_data_raw.npz" %direc)['arr_2']
tr_flux_norm = np.load("%s/tr_norm.npz" %direc)['arr_0']
tr_ivar = np.load("%s/tr_data_raw.npz" %direc)['arr_3']
tr_cont = np.load("%s/tr_cont.npz" %direc)['arr_0']
tr_SNR = np.load("%s/tr_SNRs.npz" %direc)['arr_0'][1,:]
tr_SNR = tr_SNR.astype(float)

lamost_ID = tr_ID[2].split('.')[0]
lamost_flux = tr_flux[2,:]
cont = tr_cont[2,:]
norm_flux = tr_flux_norm[2,:]
# aspcapStar-r5-v603-2M07101078+2931576.fits,4937.36572266,3.3227660656,-0.401519715786,0.243445619941
twomass = "2M07101078+2931576"
teff = "4937 K" # what are the ASPCAP uncertainties?
logg = "3.3 dex"
feh = "-0.40 dex"
apogee_file = '/home/annaho/aida41040/annaho/TheCannon/examples/example_DR12/Data/aspcapStar-r5-v603-2M07101078+2931576.fits'
apstar_file = 'apStar-r5-2M07101078+2931576.fits'
file_in = pyfits.open(apogee_file)
apogee_flux = np.array(file_in[1].data)
npixels = len(apogee_flux)
start_wl = file_in[1].header['CRVAL1']
diff_wl = file_in[1].header['CDELT1']
val = diff_wl * (npixels) + start_wl
wl_full_log = np.arange(start_wl,val, diff_wl)
wl_full = [10 ** aval for aval in wl_full_log]
apogee_wl = np.array(wl_full)

fig = plt.figure(figsize=(8,6))
props = dict(boxstyle='round', facecolor='white')

ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312)
ax2 = fig.add_subplot(313)

ranges = [[371,3192], [3697,5997], [6461,8255]]
for r in ranges:
    w = apogee_wl[r[0]:r[1]]
    f = apogee_flux[r[0]:r[1]]
    ax0.plot(w[f>0], f[f>0], c='k', linewidth=.2)
ax0.set_ylim(0.3, 1.3)
ax0.set_xlim(15100,17000)
text1 = r"$\mbox{APOGEE} (R\approx22,500):  \mbox{T}_{\mbox{eff}}, \mbox{log g}, \mbox{[Fe/H]} = 4937\, \mbox{K}, 3.3\, \mbox{dex}, -0.40\, \mbox{dex} $"
ax0.text(0.05, 0.10, text1, horizontalalignment='left', verticalalignment='bottom', transform=ax0.transAxes, bbox=props)

col = "purple"

ax1.plot(lamost_wl, lamost_flux, c='k', linewidth=.2)
ax1.plot(lamost_wl, cont, c=col)
ax1.set_xlim(3800,9100)
ax1.text(0.05, 0.10, r"LAMOST ($R \approx 1,800)$", horizontalalignment='left',
         verticalalignment='bottom', transform=ax1.transAxes, bbox=props)
ax1.set_ylabel("Flux")

ax2.plot(lamost_wl, norm_flux, c='k', linewidth=.2) 
ax2.axhline(y=1, c=col)
ax2.set_xlim(3800,9100)
ax2.text(0.05, 0.10, "Normalized LAMOST", horizontalalignment='left',
         verticalalignment='bottom', transform=ax2.transAxes, bbox=props)
ax2.set_xlabel(r"Wavelength $\lambda (\AA)$")

plt.tight_layout()
#plt.show()
plt.savefig("sample_spec.png")
