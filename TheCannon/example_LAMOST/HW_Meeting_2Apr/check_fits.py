start = 0 
stop = 3899 

SMALL = 1. / 200.

f_bar = np.zeros(len(dataset.wl))
sigma_f = np.zeros(len(dataset.wl))
nbad = np.zeros(len(dataset.wl))
for wl in range(0,len(dataset.wl)):
    array = dataset.tr_fluxes[:,wl]
    f_bar[wl] = np.median(array[array>0])
    nbad[wl] = sum(array==0)
    ngood = len(array==0)-sum(array==0)
    sigma_f[wl] = np.sqrt(np.var(array[array>0]))
    nbad[wl] = sum(array==0)
plot(dataset.wl, f_bar, alpha=0.7)
fill_between(dataset.wl, f_bar-sigma_f, f_bar+sigma_f, alpha=0.2)
scatter(dataset.wl[contmask], f_bar[contmask], c='r')
errorbar(dataset.wl[contmask], f_bar[contmask], yerr=sigma_f[contmask], c='r', fmt=None)
xlim(8600,8800)
ylim(0.8,1.1)
axhline(y=1.0)
xlabel("Wavelength (A)")
ylabel("Pseudo Cont Norm Flux")
title("Contpix identification, zoomed")
savefig("contpix_identification_9")
close()
#xlim(3800,9100) # good
#xlim(5000,5200) #bad

for jj in range(0,23):
    bad = dataset.tr_ivars[jj,:]==SMALL**2
    y = np.ma.array(dataset.tr_fluxes[jj,:], mask=bad)
    isig = np.ma.array(np.sqrt(dataset.tr_ivars[jj,:]), mask=bad)
    yerr = 1./isig
    x = np.ma.array(dataset.wl, mask=bad)
    plot(x[start:stop], y[start:stop], alpha=0.7)
    fill_between(x[start:stop], y[start:stop]-yerr[start:stop], y[start:stop]+yerr[start:stop], alpha=0.2)
    scatter(dataset.wl[start:stop][contmask[start:stop]], y[start:stop][contmask[start:stop]], c='r')
    errorbar(dataset.wl[start:stop][contmask[start:stop]], y[start:stop][contmask[start:stop]], yerr=yerr[start:stop][contmask[start:stop]], c='r', fmt=None)
    #plot(x, tr_cont[jj,:], c='k')
    xlabel('Wavelength (A)')
    ylabel('Flux')
    xlim(dataset.wl[start-10:],dataset.wl[stop+10])
    ylim(0.8,1.1)
    title('Continuum Fit')
    savefig('%s_fit.png' %jj)
    close()
