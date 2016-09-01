import pyfits
from matplotlib import rc
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
import sys
import matplotlib.gridspec as gridspec

def load_comparison():
    direc = "/Users/annaho/Data/LAMOST/Mass_And_Age"
    hdulist = pyfits.open("%s/age_vs_age_catalog.fits" %direc)
    tbdata = hdulist[1].data
    hdulist.close()
    snr = tbdata.field("snr")
    choose = snr > 0
    age = tbdata.field("cannon_age")[choose]
    age_err = tbdata.field("cannon_age_err")[choose]
    ness_age = tbdata.field("lnAge")[choose]
    ness_age_err = tbdata.field("e_logAge")[choose]
    return age, age_err, ness_age, ness_age_err

if __name__=="__main__":
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

    age, age_err, ness_age, ness_age_err = load_comparison()
    a,b,c, im = ax2.hist2d(
            np.log10(np.exp(ness_age)), age,
            bins=50, norm=LogNorm(), cmap="gray_r")
    cbar2 = plt.colorbar(im, ax=ax2, orientation='horizontal')
    cbar2.set_label("Density", fontsize=20)
    cbar2.ax.tick_params(labelsize=20)
    ax2.plot([-2,5], [-2,5], c='k')
    ax2.set_xlabel("log(Age) from APOGEE Masses", fontsize=20)
    ax2.set_ylabel("log(Age) from LAMOST C and N", fontsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.set_xlim(-0.5,1.5)
    ax2.set_ylim(-0.5,1.5)
    plt.tight_layout()
    plt.savefig("age_density.png")
    plt.show()
