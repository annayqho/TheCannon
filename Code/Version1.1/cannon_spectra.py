import numpy as np
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
    nstars, npixels, blah = cannon_set.spectra.shape
    os.system("mkdir SpectrumFits")
    contpix = list(np.loadtxt("pixtest4.txt", dtype=int, usecols=(0,), unpack=1))
