# read in all LAMOST labels

import numpy as np
from matplotlib import rc
from matplotlib import cm
import matplotlib as mpl
rc('font', family='serif')
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

direc = "/home/annaho/aida41040/annaho/TheCannon/examples"

teff = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',', 
        dtype='float', usecols=(1,), skiprows=1)
logg = np.loadtxt(
        "%s/lamost_dr2/lamost_labels_all_dates.csv" %direc, delimiter=',',         
        dtype='float', usecols=(2,), skiprows=1)

H, xedges, yedges = np.histogram2d(teff,logg,bins=50, range=[[7500,3800],[5, 0]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
cset = plt.contour(H, extent=extent)
plt.clabel(cset)
for c in cset.collections:
    c.set_linestyle("solid")
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
plt.show()
