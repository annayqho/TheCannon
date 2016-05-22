import numpy as np
from matplotlib import rc
from matplotlib import cm
import matplotlib as mpl
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

tr_IDs = np.load("%s/tr_id.npz" %direc_apogee)['arr_0']
labels_apogee = np.load("%s/tr_label.npz" %direc_apogee)['arr_0']
apogee_teff = labels_apogee[:,0]
apogee_logg = labels_apogee[:,1]
apogee_feh = labels_apogee[:,2]

