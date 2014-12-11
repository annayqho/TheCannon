# Only plots one parameter (age) instead of all four, and does not include the Pleiades.

#!/usr/bin/python 
import scipy 
import numpy 
import pickle
from numpy import * 
from scipy import ndimage
from scipy import interpolate 
from numpy import loadtxt
import os 
import numpy as np
from numpy import * 
import matplotlib 
from pylab import rcParams
from pylab import * 
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import axes
from matplotlib.pyplot import colorbar
#from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
s = matplotlib.font_manager.FontProperties()
s.set_family('serif')
s.set_size(14)
from matplotlib import rc
rc('text', usetex=False)
rc('font', family='serif')

def plotfits():
#    file_in = "self_tags.pickle"
    file_in = "self_2nd_order_tags.pickle"
    file_in2 = open(file_in, 'r') 
    params, covs_params = pickle.load(file_in2)
    sp = shape(params) 
    params = array(params)
    file_in2.close()

    filein2 = 'test14.txt' 
    filein3 = 'logAges.txt'
 
    plot_markers = ['ko', 'yo', 'ro', 'bo', 'co','k*', 'y*', 'r*', 'b*', 'c*', 'ks', 'rs', 'bs', 'cs', 'rd', 'kd', 'bd', 'rd', 'mo', 'ms' ]
    # M92, M15, M53, N5466, N4147, M13, M2, M3, M5, M107, M71, N2158, N2420, Pleaides, N7789, M67, N6819 , N188, N6791 
    t,g,feh,t_err,feh_err = loadtxt(filein2, usecols = (4,6,8,16,17), unpack =1) 
    tA,gA,fehA = loadtxt(filein2, usecols = (3,5,7), unpack =1) 
    age = loadtxt(filein3, usecols = (0,), unpack =1) 
    g_err = [0]*len(g) 
    age_err = [0]*len(g) 
    g_err = array(g_err)
    diffT = abs(array(t) - array(tA) ) 
    pick = diffT < 600. 
    t,g,feh,t_err,g_err,feh_err = t[pick], g[pick], feh[pick], t_err[pick], g_err[pick], feh_err[pick] 
    age = array(age)
    age_err = array(age_err) 
    age, age_err = age[pick] , age_err[pick] 
    
    # Indices of the Pleiades stars
    pleiades = []

    a = open(filein2) 
    al = a.readlines() 
    names = []
    for each in al:
      names.append(each.split()[1]) 
    names = array(names)
    names = names[pick]

    # Find the Pleiades stars...
    for i in range(0, len(names)):
        if names[i] == "Pleiades":
            pleiades.append(i)

    # Remove the Pleiades stars from the names
    # names = delete(names, pleiades)

    # Remove the Pleiades stars from the literature values
    t,g,feh,t_err,g_err,feh_err, age, age_err = delete(t, pleiades), delete(g, pleiades),delete(feh, pleiades),delete(t_err, pleiades),delete(g_err, pleiades),delete(feh_err, pleiades), delete(age, pleiades), delete(age_err, pleiades)

    unames = unique(names) 
    starind = arange(0,len(names), 1) 
    name_ind = [] 
    names = array(names) 
    for each in unames:
        takeit = each == names
        name_ind.append(np.int(starind[takeit][-1]+1. ) )
    cluster_ind = [0] + list(sort(name_ind))# + [len(al)]
   
    params = array(params)[pick]
    covs_params = array(covs_params)[pick]

    # Remove the Pleiades stars from the result values
    params, covs_params = delete(params, pleiades, axis=0), delete(covs_params, pleiades, axis=0)

    sp2 = shape(params) 
    sp3 = len(t) 
    #covs_params = np.linalg.inv(icovs_params) 
    rcParams['figure.figsize'] = 12.0, 10.0
    
    # Remove the Pleiades from cluster_ind as well as the plot_markers
    index = cluster_ind.index(438)
    cluster_ind.pop(index)
    plot_markers.pop(index-1)
    #cluster_ind = delete(cluster_ind, index)
    #plot_markers = delete(plot_markers, index) 

    fig, ax = pyplot.subplots()
    ax.set_title("log(Age) Results from the Cannon")
    params_labels = [params[:,0], params[:,1], params[:,2] , params[:,3], covs_params[:,0,0]**0.5, covs_params[:,1,1]**0.5, covs_params[:,2,2]**0.5 , covs_params[:,3,3]**0.5]
    cval = ['k', 'b', 'r', 'c'] 
    input_ASPCAP = [t, g, feh, age, t_err, g_err, feh_err, age_err] 
    listit_1 = [0,1,2,3]
    listit_2 = [1,0,0,0]
    labels = ['teff', 'logg', 'Fe/H', 'age' ]
    for i in range(0,len(cluster_ind)-1): 
        indc1 = cluster_ind[i]
        indc2 = cluster_ind[i+1]
        datay = params_labels[3]
        datax = input_ASPCAP[3]
        pick = logical_and(g[indc1:indc2] > 0, logical_and(t_err[indc1:indc2] < 300, feh[indc1:indc2] > -4.0) ) 
        ax.plot(input_ASPCAP[3][indc1:indc2][pick], params_labels[3][indc1:indc2][pick], plot_markers[i]) 
    ax.text(10.0,7.0,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[3+4]),2)),fontsize = 14)

    ax.plot([-5,25], [-5,25], linewidth = 1.5, color = 'k' ) 
    ax.set_xlim(7,11)  # for log ages
    ax.set_ylim(6,12) #for log ages 
    ax.set_ylabel("log(Age [Gyr])", fontsize = 14,labelpad = 5) 
    ax.set_xlabel("Literature log(Ages)", fontsize = 14,labelpad = 10) 
    prefix = "/home/annaho/AnnaCannon/test_self"
    savefig(fig, prefix, transparent=False, bbox_inches='tight', pad_inches=0.5, orientation='landscape')
    return 

def savefig(fig, prefix, **kwargs):
    for suffix in (".eps", ".png"):
        print "writing %s" % (prefix + suffix)
        fig.savefig(prefix + suffix, **kwargs)

if __name__ == "__main__": #args in command line 
    wl1,wl2,wl3,wl4,wl5,wl6 = 15392, 15697, 15958.8, 16208.6, 16120.4, 16169.5 
    plotfits() 

