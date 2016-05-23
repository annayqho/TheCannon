import numpy as np
from matplotlib import rc
from TheCannon import model
from TheCannon import dataset
from lamost import load_spectra, load_labels

rc('text', usetex=True)
rc('font', family='serif')

tr_ID = np.loadtxt("example_PS1/ps_colors_ts_overlap.txt", 
                   usecols=(1,), dtype='str', delimiter=',')

dir_dat = "example_LAMOST/Data_All"
tr_IDs, wl, tr_flux, tr_ivar = load_spectra(dir_dat, tr_ID)

label_file = "apogee_dr12_labels.csv"
all_labels = load_labels(label_file, tr_IDs)
teff = all_labels[:,0]
logg = all_labels[:,1]
mh = all_labels[:,2]
alpha = all_labels[:,3]
tr_label = np.vstack((teff, logg, mh, alpha)).T

data = dataset.Dataset(
        wl, tr_IDs, tr_flux, tr_ivar, tr_label, 
        tr_IDs, tr_flux, tr_ivar)
data.set_label_names(['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]'])
data.continuum_normalize_gaussian_smoothing(L=50)

# get colors

colors = np.loadtxt("example_PS1/ps_colors_ts_overlap.txt", 
                    usecols=(2,4,6,8), dtype='float', delimiter=',')
errors = np.loadtxt("example_PS1/ps_colors_ts_overlap.txt", 
                    usecols=(3,5,7,9), dtype='float', delimiter=',') 
ivars = 1./ errors**2
colors = colors[np.argsort(tr_ID)]
ivars = ivars[np.argsort(tr_ID)]
ivars = ivars * 1e15

# add another column to the tr_flux, tr_ivar, test_flux, test_ivar

logwl = np.log(data.wl)
delta = logwl[1]-logwl[0]
toadd = logwl[-1]+delta*np.arange(1,5)
new_logwl = np.hstack((logwl, toadd))
data.wl = np.exp(new_logwl)
data.tr_flux = np.hstack((data.tr_flux, colors))
data.test_flux = data.tr_flux
data.tr_ivar = np.hstack((data.tr_ivar, ivars))
data.test_ivar = data.tr_ivar

# train model
m = model.CannonModel(2) # 2 = quadratic model
m.fit(data)
m.infer_labels(data)
# data.diagnostics_1to1()

def scatter(i):
    return np.std(data.tr_label[:,i]-data.test_label_vals[:,i])

def bias(i):
    return np.mean(data.tr_label[:,i]-data.test_label_vals[:,i])

for i in range(0,4):
    print(scatter(i), bias(i))
