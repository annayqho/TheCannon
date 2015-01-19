import numpy as np
import os
from dataset import Dataset

def draw_spectra(label_vector, model, test_set):
    coeffs_all, covs, scatters, chis, chisqs, pivots = model
    nstars = len(test_set.IDs)
    cannon_spectra = np.zeros(test_set.spectra.shape)
    cannon_spectra[:,:,0] = test_set.spectra[:,:,0]
    for i in range(nstars):
        x = label_vector[:,i,:]
        spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
        cannon_spectra[i,:,1]=spec_fit
    cannon_set = Dataset(IDs=test_set.IDs, SNRs=test_set.SNRs, 
            spectra=cannon_spectra, label_names = test_set.label_names, 
            label_values = test_set.label_values)
    return cannon_set

def diagnostics(cannon_set, test_set):
    # Overplot original spectra with best-fit spectra
    os.system("mkdir SpectrumFits")
    #contpix = list(np.loadtxt("pixtest4.txt", dtype=int, usecols=(0,), unpack=1))
    #contmask = np.zeros(8575, dtype=bool)
    #contmask[contpix] = 1
    nstars = fitted_spec.shape[0]
    for i in range(0,1):
        print "Star %s" %i
        ID = cannon_set.IDs[i]
        spec_orig = test_set.spectra[i,:,1]
        spec_fit = cannon_set.spectra[i,:,1]
        bad = np.logical_or(spec_orig == 0, spec_orig == 1)
        keep = np.invert(bad)
        pixels = test_set.spectra[i,:,0]
        fig,axarr = plt.subplots(2)
        ax1 = axarr[0]
        ax1.scatter(pixels[keep], spec_fit[keep], label="Cannon Spectrum", linewidth=0.5, color='r')
        ax1.scatter(pixels[keep], spec_orig[keep], label="Orig Spectrum",
                linewidth=0.5, alpha=0.7, color='b')
        ax1.set_xlabel(r"Wavelength $\lambda (\AA)$")
        ax1.set_ylabel("Normalized flux")
        ax1.set_title("Spectrum Fit: %s" %ID)
        ax1.legend(loc='lower center', fancybox=True, shadow=True)
        ax2 = axarr[1]
        ax2.scatter(spec_orig[keep], spec_fit[keep])
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        lims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
        ax2.plot(lims, lims, 'k-', alpha=0.75)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.set_xlabel("Orig Fluxes")
        ax2.set_ylabel("Fitted Fluxes")
        filename = "Star%s.png" %i
        print "Diagnostic plot: fitted vs. original spec"
        print "Saved as %s" %filename
        fig.savefig("SpectrumFits/"+filename)
        plt.close(fig)
