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

    fn = 'apokasc_all_ages.txt' 
    
    plot_markers = ['ko', 'yo', 'ro', 'bo', 'co','k*', 'y*', 'r*', 'b*', 'c*', 'ks', 'rs', 'bs', 'cs', 'rd', 'kd', 'bd', 'rd', 'mo', 'ms' ]
    names = loadtxt(fn, dtype='string', usecols = [0], unpack = 1)
    t,g,feh,age = loadtxt(fn, usecols = (6, 8, 4, 10), unpack = 1)
    t_err, g_err, feh_err = loadtxt(fn, usecols = (7, 9, 5), unpack = 1)
    age_err = [0]*len(g) 
    #age = array(age)
    age_err = array(age_err)

    # We only want the elements corresponding to the stars in the directory

    pick = []

    dir = '/home/annaho/AnnaCannon/Code/Maries_Data/'
    file_list = []
    for file in os.listdir(dir):
        if file.startswith("aspcapStar") and file.endswith(".fits"):
            file_list.append('%s%s' %(dir,file))

    for file in file_list:
        starname = file.split('-')[2].split('.')[0]
        searchfor = 'J%s' %(starname[2:]) # listed as J in the apokasc file instead of 2M
        index = where(names==searchfor)[0][0]
        pick.append(index)

    t,g,feh,age,t_err,g_err,feh_err,age_err = t[pick], g[pick], feh[pick], age[pick], t_err[pick], g_err[pick], feh_err[pick], age_err[pick]

    params = array(params)
    covs_params = array(covs_params)
    sp2 = shape(params) 
    sp3 = len(t) 
    rcParams['figure.figsize'] = 12.0, 10.0
    fig, temp = pyplot.subplots(3,1, sharex=False, sharey=False)
    fig = plt.figure() 
    ax = fig.add_subplot(111, frameon = 0 ) 
    ax.set_ylabel("The Cannon", labelpad = 40, fontsize = 20 ) 
    ax.tick_params(labelcolor= 'w', top = 'off', bottom = 'off', left = 'off', right = 'off' ) 
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    params_labels = [params[:,0], params[:,1], params[:,2] , params[:,3], covs_params[:,0,0]**0.5, covs_params[:,1,1]**0.5, covs_params[:,2,2]**0.5 , covs_params[:,3,3]**0.5]
    cval = ['k', 'b', 'r', 'c'] 
    input_ASPCAP = [t, g, feh, age, t_err, g_err, feh_err, age_err] 
    listit_1 = [0,1,2,3]
    listit_2 = [1,0,0,0]
    axs = [ax1,ax2,ax3,ax4]
    labels = ['teff', 'logg', 'Fe/H', 'age' ]
    for ax, num,num2,label1,x1,y1 in zip(axs, listit_1,listit_2,labels, [4800,3.0,0.3,0.3], [3400,1,-1.5,5]): 
        #pick = logical_and(g[indc1:indc2] > 0, logical_and(t_err[indc1:indc2] < 300, feh[indc1:indc2] > -4.0) ) 
        #cind = array(input_ASPCAP[1][indc1:indc2][pick]) 
        #cind = array(input_ASPCAP[num2][indc1:indc2][pick]).flatten() 
        print shape(input_ASPCAP)
        print shape(params_labels)
        ax.scatter(input_ASPCAP[num], params_labels[num]) 
        ax.errorbar(input_ASPCAP[num], params_labels[num],xerr=input_ASPCAP[num+4],ls='',zorder=0, fmt = None,elinewidth = 1,capsize = 0)
    ax1.text(5400,3700,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[0+4]),2)),fontsize = 14) 
    ax2.text(3.9,1,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[1+4]),2)),fontsize = 14) 
    ax3.text(-0.3,-2.5,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[2+4]),2)),fontsize = 14) 
    ax4.text(13.0, 5.0,"y-axis, $<\sigma>$ = "+str(round(mean(params_labels[3+4]),2)),fontsize = 14)

    ax1.plot([0,6000], [0,6000], linewidth = 1.5, color = 'k' ) 
    ax2.plot([0,5], [0,5], linewidth = 1.5, color = 'k' ) 
    ax3.plot([-3,2], [-3,2], linewidth = 1.5, color = 'k' ) 
    ax4.plot([-5,25], [-5,25], linewidth = 1.5, color = 'k' ) 
    #ax4.set_xlim(7,11)  # for log ages
    ax4.set_xlim(0, 15) 
    ax4.set_ylim(-3, 20) 
    #ax4.set_ylim(6,12) #for log ages 
    ax1.set_xlim(3900, 5300) 
    ax1.set_ylim(1000,6000)
    ax1.set_ylim(3500,6000)
    ax2.set_xlim(1, 4)
    ax3.set_xlim(-1.3, 0.8) 
    ax1.set_xlabel("ASPCAP Teff, [K]", fontsize = 14,labelpad = 5) 
    ax1.set_ylabel("Teff, [K]", fontsize = 14,labelpad = 5) 
    ax2.set_xlabel("ASPCAP logg, [dex]", fontsize = 14,labelpad = 5) 
    ax2.set_ylabel("logg, [dex]", fontsize = 14,labelpad = 5) 
    ax3.set_xlabel("ASPCAP [Fe/H], [dex]", fontsize = 14,labelpad = 5) 
    ax3.set_ylabel("[Fe/H], [dex]", fontsize = 14,labelpad = 5) 
    ax4.set_ylabel("Age [Gyr]", fontsize = 14,labelpad = 5) 
    ax4.set_xlabel("ASPCAP ages [Gyr]", fontsize = 14,labelpad = 10) 
    ax2.set_ylim(0,5)
    ax3.set_ylim(-2,1) 
    # attach lines to plots
    fig.subplots_adjust(hspace=0.22)
    #prefix = "/Users/ness/Downloads/Apogee_Raw/calibration_apogeecontinuum/documents/plots/fits_3_self_cut"
    prefix = "/home/annaho/AnnaCannon/Maries_Stars"
    savefig(fig, prefix, transparent=False, bbox_inches='tight', pad_inches=0.5)
    print sp, sp2, sp3
    return 

def savefig(fig, prefix, **kwargs):
    for suffix in (".eps", ".png"):
        print "writing %s" % (prefix + suffix)
        fig.savefig(prefix + suffix, **kwargs)

if __name__ == "__main__": #args in command line 
    wl1,wl2,wl3,wl4,wl5,wl6 = 15392, 15697, 15958.8, 16208.6, 16120.4, 16169.5 
    plotfits() 

