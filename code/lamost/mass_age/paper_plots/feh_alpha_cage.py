""" Copied from Melissa's Mass & Age paper """

import pyfits
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np

cmap = plt.cm.RdYlBu_r

def load_data():
    direc = "/Users/annaho/Data/LAMOST/Mass_And_Age"
    hdulist = pyfits.open("%s/catalog_paper.fits" %direc)
    tbdata = hdulist[1].data
    hdulist.close()
    snr = tbdata.field("snr")
    chisq = tbdata.field("chisq")
    teff = tbdata.field("cannon_teff")
    in_martig_range = tbdata.field("in_martig_range")
    choose = np.logical_and(in_martig_range, snr > 80)
    print(sum(choose))
    mh = tbdata.field("cannon_mh")[choose]
    afe = tbdata.field("cannon_am")[choose]
    age = tbdata.field("cannon_age")[choose]
    age_err = tbdata.field("cannon_age_err")[choose]
    return mh, afe, age, age_err

if __name__=="__main__":
    mh, am, age, age_err = load_data()
    nx = 20
    ny = 10
    nmin = 20

    total_age, xedges, yedges, im2 = plt.hist2d(
            mh, am, bins=[nx,ny], weights = age * age_err, 
            normed = False, cmin=nmin)
    total_err, xedges, yedges, im3 = plt.hist2d(
            mh, am, bins=[nx,ny], weights = age_err, 
            normed = False, cmin=nmin)
    mean_age = total_age / total_err

    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    fig = plt.figure(figsize=(8,4))
    # Returns the number of samples in each bin
    #count, xedges, yedges, im1 = plt.hist2d(
    #        mh, am, bins=[nx,ny], normed = False, cmin = 50)
    # Sum of the weighted ages in each bin
    # Sum of the errors in each bin
    # Sum of the differences from the mean, squared
    #age_sqr, xedges, yedges, im3 = plt.hist2d(
    #        mh, am, bins=[nx,ny], weights = (age)**2, normed = False, cmin=50)

    im = plt.imshow(
            10**mean_age.T, interpolation='nearest', aspect='auto', 
            origin='lower', cmap=cmap, vmin=0, vmax=12, 
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))
    cbar1 = plt.colorbar(im, orientation='vertical')
    cbar1.set_label("Mean Age [Gyr]", fontsize=20)
    cbar1.ax.tick_params(labelsize=20)
    plt.xlim(-0.8, 0.35)
    plt.ylim(-0.05,0.3)
    plt.ylabel(r"[$\alpha$/M]", fontsize=20)
    plt.xlabel("[M/H]", fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    plt.savefig("mh_am_cage.png")
    #plt.show()
