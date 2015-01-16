    # Histogram of the chi squareds of the fits
    plt.hist(chis)
    dof = len(pixels) - nlabels
    dofline = plt.axvline(x=dof, color='b', linestyle='dotted', 
            label="DOF = npixels - nlabels")
    plt.legend()
    plt.title("Distribution of Chi Squareds of the Model Fit")
    plt.ylabel("Count")
    plt.xlabel("Chi Squared")
    filename = "modelfit_chisqs.png"
    print "Diagnostic plot: histogram of the chi squareds of the fit"
    print "Saved as %s" %filename
    plt.savefig(filename)
    plt.close()

    # Overplot original spectra with the best-fit spectra
    # We have: the label vector x, and the coefficient vector coeffs_all
    # f_lambda = np.dot(x, coeff)
    # Perform this for each star and plot the spectrum
    nstars = label_vector.shape[1]
    npixels = training_set.spectra.shape[1]
    os.system("mkdir SpectrumFits")
    contpix = list(np.loadtxt("pixtest4.txt", dtype=int, usecols=(0,), unpack=1))
    contmask = np.zeros(8575, dtype=bool)
    contmask[contpix] = 1
    fitted_spec = np.zeros((nstars,npixels))
    for i in range(nstars):
        print "star %s" %i
        x = label_vector[:,i,:]
        ID = training_set.IDs[i]
        spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
        fitted_spec[i,:] = spec_fit
        spec_orig = training_set.spectra[i,:,1]
        bad1 = spec_orig == 0
        bad2 = contmask
        keep = np.invert(bad1 | bad2)
        fig, axarr = plt.subplots(2)
        ax1 = axarr[0]
        ax1.plot(pixels, spec_fit, label="Cannon Spectrum", linewidth=0.5)
        ax1.plot(pixels[keep], spec_orig[keep], label="Orig Spectrum", 
                linewidth=0.5, alpha=0.7)
        ax1.set_xlabel(r"Wavelength $\lambda (\AA)$")
        ax1.set_ylabel("Normalized flux")
        ax1.set_title("Spectrum Fit: %s" %ID)
        ax1.legend(loc='lower center', fancybox=True, shadow=True)
        ax2 = axarr[1]
        ax2.scatter(spec_orig[keep], spec_fit[keep])
        ax2.set_xlabel("Orig Fluxes")
        ax2.set_ylabel("Fitted Fluxes")
        filename = "Star%s.png" %i
        print "Diagnostic plot: fitted vs. original spec"
        print "Saved as %s" %filename
        fig.savefig("SpectrumFits/"+filename)
        plt.close(fig)
