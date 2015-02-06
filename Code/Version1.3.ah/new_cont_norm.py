def identify_continuum(lambdas, spectra):
    """Identifies continuum pixels."""

    f_bar = np.median(spectra[:,:,0], axis=0)
    sigma_f = np.var(spectra[:,:,0], axis=0)
    # f_bar == 0
    cont1 = f_bar == 0
    # f_bar ~ 1...
    f_cut = 0.001
    cont2 = np.abs(f_bar-1)/1 < f_cut
    # sigma_f << 1...
    sigma_cut = 0.005
    cont3 = sigma_f < sigma_cut
    cont = np.logical_or(cont1, np.logical_and(cont2, cont3))
    #plot(lambdas, f_bar)
    #errorbar(lambdas[cont], f_bar[cont], yerr=sigma_f[cont], fmt='ko')
    return lambdas[cont]


