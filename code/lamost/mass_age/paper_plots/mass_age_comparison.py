import pyfits
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
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
    mass_raw = tbdata.field("cannon_mass")
    choose = np.logical_and(snr > 30, mass_raw > 0)
    mass = np.log10(tbdata.field("cannon_mass")[choose])
    age = tbdata.field("cannon_age")[choose]
    age_err = tbdata.field("cannon_age_err")[choose]
    mass_err = tbdata.field("cannon_mass_err")[choose]
    ness_age = np.log10(np.exp(tbdata.field("lnAge")[choose]))
    ness_age_err = tbdata.field("e_logAge")[choose]
    ness_mass = np.log10(np.exp(tbdata.field("lnM")[choose]))
    ness_mass_err = tbdata.field("e_logM")[choose]
    return mass, age, ness_mass, ness_age


def plot(ax, x, y, xlabel, ylabel, axmin, axmax):
    a,b,c, im = ax.hist2d(
            x, y,
            bins=40, norm=LogNorm(), cmap="gray_r",
            range=([axmin,axmax],[axmin,axmax]))
    ax.plot([axmin,axmax], [axmin,axmax], c='k')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5))
    return im


if __name__=="__main__":
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    mass, age, ness_mass, ness_age = load_comparison()
    im = plot(
            ax1, ness_mass, mass,
            "log(Mass) from APOGEE", "log(Mass) from LAMOST C and N",
            -0.3, 0.5)
    im = plot(
            ax2, ness_age, age, 
            "log(Age) from APOGEE Masses", "log(Age) from LAMOST C and N",
            -0.5,1.5)
    cbar = plt.colorbar(im)
    cbar.set_label("Density", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    #plt.tight_layout()
    #plt.savefig("age_density.png")
    plt.show()
