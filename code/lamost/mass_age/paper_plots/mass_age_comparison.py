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


def plot(ax, x, y, xlabel, ylabel, axmin, axmax, text):
    a,b,c, im = ax.hist2d(
            x, y,
            bins=40, norm=LogNorm(), cmap="gray_r",
            range=([axmin,axmax],[axmin,axmax]))
    ax.plot([axmin,axmax], [axmin,axmax], c='k')
    #props = dict(boxstyle='round', facecolor='white', pad=0.1)
    ax.text(
            0.05, 0.8, text, 
            horizontalalignment='left', verticalalignment='bottom', 
            transform=ax.transAxes, fontsize=25)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    #cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    #cbar.set_label("Density", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5))
    return im


def plot_marginal_hist(ax, val, val_min, val_max):
    ax.hist(
            val, bins=30, orientation='horizontal',
            range=(val_min, val_max), color='black', alpha=0.5,
            histtype='stepfilled')
    ax.set_xlabel("Number of Objects", fontsize=16)
    ax.set_ylim(val_min, val_max)
    ax.tick_params(axis='x', labelsize=20)
    ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5))
    ax.set_yticklabels([])
    return ax


if __name__=="__main__":
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
    mass, age, ness_mass, ness_age = load_comparison()
    mass_min = -0.3
    mass_max = 0.5
    age_min = -0.5
    age_max = 1.5
    im = plot(
            ax1, ness_mass, mass,
            "Via APOGEE Spectra", "Via LAMOST C and N",
            mass_min, mass_max, r"log(Mass/M${}_\odot$)")
    im = plot(
            ax2, ness_age, age, 
            "Via APOGEE Spectroscopic Mass + Isochrones", "Via LAMOST C and N",
            age_min,age_max, r"log(Age/Gyr)")
    plt.subplots_adjust(right=0.6)
    # left, bottom, width, height
    new_ax = fig.add_axes([0.65, 0.65, 0.25, 0.25])
    hist1 = plot_marginal_hist(new_ax, mass, mass_min, mass_max)
    new_ax = fig.add_axes([0.65, 0.21, 0.25, 0.25])
    hist2 = plot_marginal_hist(new_ax, age, age_min, age_max)
    #plt.tight_layout()
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.set_label("Density", fontsize=16)
    #plt.savefig("mass_age_comparison.png")
    plt.show()
