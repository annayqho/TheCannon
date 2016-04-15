import numpy as np
import pyfits
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

apstar_file = 'apStar-r5-2M07101078+2931576.fits'

file_in = pyfits.open(apstar_file)
flux = file_in[1].data[0,:]
err = file_in[2].data[0,:]
bad = err > 4.5 
flux_masked = flux[~bad] 
npixels = len(flux)
start_wl = file_in[1].header['CRVAL1']
diff_wl = file_in[1].header['CDELT1']
val = diff_wl * (npixels) + start_wl
wl_full_log = np.arange(start_wl,val, diff_wl)
wl_full = [10 ** aval for aval in wl_full_log]
apogee_wl = np.array(wl_full)
wl = apogee_wl[~bad]

plot(wl, flux_masked, c='k', linewidth=0.5)
title("APOGEE Combined Spectrum", fontsize=30)
ylabel("Flux", fontsize=30)
xlabel(r"Wavelength (Angstroms)", fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
savefig("apogee_combined_spec.png")

lamost_wl = np.load("../examples/test_training_overlap/tr_data_raw.npz")['arr_1']
tr_flux_norm = np.load("../examples/test_training_overlap/tr_norm.npz")['arr_0']
tr_flux = np.load("../examples/test_training_overlap/tr_data_raw.npz")['arr_2']
tr_cont = np.load("../examples/test_training_overlap/tr_cont.npz")['arr_0']
lamost_flux_norm = tr_flux_norm[2,:]
lamost_flux = tr_flux[2,:]
cont = tr_cont[2,:]
plot(lamost_wl, lamost_flux, c='k', linewidth=0.5)
xlim(3800, 9100)
axhline(y=1, c='r')
title("LAMOST Normalized Spectrum", fontsize=30)
ylabel("Flux", fontsize=30)
xlabel(r"Wavelength (Angstroms)", fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
savefig("lamost_normalized_spec.png")
