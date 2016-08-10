# DWH did a bunch of stuff on my computer

from TheCannon import model
from TheCannon import dataset
wl = np.load("wl.npz")['arr_0']
m = model.CannonModel(2)
m.coeffs = np.load("./coeffs.npz")['arr_0']
m.scatters = np.load("./scatters.npz")['arr_0']
m.chisqs = np.load("./chisqs.npz")['arr_0']
m.pivots = np.load("./pivots.npz")['arr_0']
ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
test_labels = np.load("test_results_0.npz")['arr_0']
ds.test_label_vals = test_labels
m.infer_spectra(ds)
plot(wl, m.model_spectra[0,:], c='k')
# show it's log scale: (wl[1:] - wl[:-1]) / (wl[1:] + wl[:-1])
# km/s: 299792.458 * (wl[1:] - wl[:-1]) / (wl[1:] + wl[:-1])
deltav = np.median(299792.458 * (wl[1:] - wl[:-1]) / (wl[1:] + wl[:-1]))
foo = m.model_spectra[0,:]
dfdv = (foo[1:] - foo[:-1]) / deltav
# derivative wrt wavelength plot(dfdv)
np.dot(dfdv, dfdv)
np.dot(dfdv, dfdv) * 350 * 350
res = 1. / np.sqrt(np.dot(dfdv, dfdv) * 350 * 350)
# res = 52 km/s from a LAMOST spectrum
