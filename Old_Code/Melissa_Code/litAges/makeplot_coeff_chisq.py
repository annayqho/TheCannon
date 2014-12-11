#!/usr/bin/python
import pickle
import numpy
from numpy import savetxt
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
s = matplotlib.font_manager.FontProperties()
s.set_family('serif')
s.set_size(14)
from matplotlib import rc
rc('text', usetex=False)
rc('font', family='serif')
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
s = matplotlib.font_manager.FontProperties()
s.set_family('serif')
from matplotlib import rcParams
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
s = matplotlib.font_manager.FontProperties()
majorLocator   = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(5)
yminorLocator   = MultipleLocator(10)
yminorLocator2   = MultipleLocator(25)
xminorLocator   = MultipleLocator(5)
yminorLocator   = MultipleLocator(5)
ymajorLocator   = MultipleLocator(50)
xmajorLocator   = MultipleLocator(10)
rcParams['figure.figsize'] = 15.0, 10.0

from fitspectra_ages import get_goodness_fit
f1 = 'normed_data'
f2 = 'starsin_SFD_Pleiades.txt'
f3 = 'coeffs_2nd_order.pickle'

chisq = get_goodness_fit(f1, f2, f3) # chisq.shape = (553, 8575)

#x, median_y, t_y, g_y,feh_y,chi_y = loadtxt('data_test.txt', usecols = (0,1,2,3,4,5), unpack =1) 
#fig1 = pyplot.figure()
#ax0 = fig1.add_subplot(111)
fig, ax = plt.subplots()

file_in = 'coeffs_2nd_order.pickle'
file_in2 = open(file_in, 'r') 
dataall, metaall, labels, offsets, coeffs, covs, scatters, chis, chisqs = pickle.load(file_in2)
file_in2.close()

#sortindx = 2
#sortname = ["Teff", "logg", "Fe/H"]
#index_use = np.argsort(metaall[:,sortindx])
#ax.set_title("per-pixel scaled residuals (coeff); spectra ordered by %s" % (sortname[sortindx]),fontsize = 20 ) 
ax.set_title("per-pixel reduced chi squared")
ax.set_xlabel("Wavelength-direction pixel number",fontsize = 20,labelpad = 10 ) 
ax.set_ylabel("Star Number",fontsize = 20) 
#print "Ordered by %s" % (sortname[sortindx]) 

#a = open('starsin_SFD_Pleiades.txt', 'r')
#al = a.readlines()
#names = []
#for each in al:
#    names.append(each.split()[1]) 
#unames = np.unique(names) 
#starind = np.arange(0,len(names), 1) 
#name_ind = [] 
#names = np.array(names) 
#for each in unames:
#    takeit = each == names 
#    name_ind.append(starind[takeit][-1]+1. ) 

image = chisq.T
#b = np.insert(a, 3, values=0, axis=1)
test = ax.imshow(image.T, cmap=plt.cm.bwr_r, interpolation="nearest", vmin = 0, vmax = 1,aspect = 'auto',origin = 'lower')
cb = fig.colorbar(test) 
#cb.set_label("arcsinh($\chi$)", fontsize = 20 ) 
cb.set_label("chisq value", fontsize = 20 )
#fig.show()
fig.savefig('chisq_map_findN6791.eps', transparent=True, bbox_inches='tight', pad_inches=0)
