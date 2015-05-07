import pickle
import numpy as np

wl, tr_flux, tr_ivar, tr_label, test_flux, test_ivar = pickle.load(
        open('lamost_data.p', 'r'))

# MAKE S/N AND NPIX ARRAYS FOR THE STARS
tr_nstars = tr_flux.shape[0]
test_nstars = test_flux.shape[0]
npix = tr_flux.shape[1]

tr_snr = tr_flux * np.sqrt(tr_ivar)
bad1 = tr_snr == 0
tr_snr = np.ma.array(tr_snr, mask=bad1)
tr_snr = np.ma.median(tr_snr, axis=1)
test_snr = test_flux * np.sqrt(test_ivar)
bad2 = test_snr == 0
test_snr = np.ma.array(test_snr, mask=bad2)
test_snr = np.ma.median(test_snr, axis=1)
snr = np.concatenate((tr_snr, test_snr))

tr_npix = np.zeros(tr_nstars)
test_npix = np.zeros(test_nstars)
spec_npix_tr = np.zeros(npix)
spec_npix_test = np.zeros(npix)


for jj in range(0, tr_nstars):
    tr_npix[jj] = np.count_nonzero(tr_ivar[jj,:])

for jj in range(0, test_nstars):
    test_npix[jj] = np.count_nonzero(test_ivar[jj,:])

for jj in range(0, npix):
    spec_npix_tr[jj] = np.count_nonzero(tr_ivar_cut[:,jj]) 
    spec_npix_test[jj] = np.count_nonzero(test_ivar_cut[:,jj])

cut =200 
tr_ivar_cut = tr_ivar[tr_snr>cut, :]
test_ivar_cut = test_ivar[test_snr>cut,:]


# plot f x sqrt(ivar) across the spectrum: should get a Gaussian around unity

snrs = dataset.tr_fluxes * np.sqrt(dataset.tr_ivars)
norm_snrs = snrs / dataset.tr_SNRs[:,None]
bad = norm_snrs == 0
norm_snrs = np.ma.array(norm_snrs, mask=bad)

# investigate the fraction of bad pixels in each star
badcount = np.sum(bad, axis=1)
badfrac = badcount/float(norm_snrs.shape[1])

bad = norm_snrs[0,:] == 0
y = np.ma.array(norm_snrs[0,:], mask=bad)
plot(dataset.wl, y)
xlim(3500,9200)
title("SNR across spectrum, no pix masked")
xlabel("Wavelength (A)")
ylabel("Normalized SNR")

imshow(norm_snrs, aspect='auto')
colorbar()
xlabel("Pixel")
ylabel("Star")
title("Normalized SNR")
title("Normalized SNR, Raw Data")
savefig("normsnr_allpix.png")

from random import randint
f,axarr=subplots(4,4)
for i in range(0,4):
    for j in range(0,4):
        print(i,j)
        axarr[i,j].plot(norm_snrs[randint(0,800)].compressed())
        axarr[i,j].set_xlim(-1,3.8)
savefig("normsnr_16stars.png")
