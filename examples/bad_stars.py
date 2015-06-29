import pickle
import random
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# STEP 1: DATA MUNGING
import glob
allfiles = glob.glob("example_LAMOST/Data_All/*fits")
allfiles = np.char.lstrip(allfiles, 'example_LAMOST/Data_All/')
tr_ID = np.loadtxt("tr_files.txt", dtype=str)
test_ID = np.setdiff1d(allfiles, tr_ID)
from lamost import load_spectra, load_labels
dir_dat = "example_LAMOST/Data_All"
tr_IDs, wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_ID)
label_file = "reference_labels.csv"
tr_label = load_labels(label_file, tr_ID)
test_IDs, wl, test_flux, test_ivar = load_spectra(dir_dat, test_ID)
good = np.logical_and(tr_label[:,0] > 0, tr_label[:,2]>-5)
tr_IDs = tr_IDs[good]
tr_flux = tr_flux[good]
tr_ivar = tr_ivar[good]
tr_label = tr_label[good]
from TheCannon import dataset
dataset = dataset.Dataset(
    wl, tr_IDs, tr_flux, tr_ivar, tr_label, test_IDs, test_flux, test_ivar)
dataset.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])
cont = pickle.load(open("cont.p", "r"))
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
                dataset.continuum_normalize(cont)
tr_cont, test_cont = cont
tr_cont[tr_cont==0] = dataset.tr_flux[tr_cont==0]
test_cont[test_cont==0] = dataset.test_flux[test_cont==0]
cont = tr_cont, test_cont
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
                dataset.continuum_normalize(cont)
dataset.tr_flux = norm_tr_flux
dataset.tr_ivar = norm_tr_ivar
dataset.test_flux = norm_test_flux
dataset.test_ivar = norm_test_ivar
label_file = 'reference_labels.csv'
# for each test ID, find its index in label_file IDs
ids = np.loadtxt(label_file, usecols=(0,), dtype=str, delimiter=',')
inds = [np.where(ids==test_ID_val) for test_ID_val in test_ID]
names = ['T_{eff}', '\log g', '[Fe/H]', '[\\alpha/Fe]']
lims = [[3900,6000], [0,5], [-2, 1], [-0.1,0.4]]
#id,teff,logg,feh,alpha,snr
teff = np.loadtxt(label_file, usecols=(1,), dtype=float, delimiter=',')
logg = np.loadtxt(label_file, usecols=(2,), dtype=float, delimiter=',')
feh = np.loadtxt(label_file, usecols=(3,), dtype=float, delimiter=',')
alpha = np.loadtxt(label_file, usecols=(4,), dtype=float, delimiter=',')
apogee_label_vals = np.vstack(
                (teff[inds].flatten(), logg[inds].flatten(), feh[inds].flatten(), alpha[inds].flatten())).T
choose = np.logical_and(orig<0.15, cannon>0.18)
c=np.zeros(len(choose))
c[choose] = 'r'
c=np.zeros(len(choose), dtype='str')
c[choose] = 'r'
c[~choose] = 'k'
scatter(orig, cannon, c=c)

