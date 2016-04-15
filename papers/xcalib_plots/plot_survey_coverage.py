#!/usr/bin/env python
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.colors import LogNorm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import pyfits

# import the data
hdulist = pyfits.open("../make_lamost_catalog/lamost_catalog_full.fits")
tbdata = hdulist[1].data
# cols = hdulist[1].columns
# cols.names
ra_lamost = tbdata.field('ra')
dec_lamost = tbdata.field('dec')
am_lamost = tbdata.field("cannon_a_k")
rmag_lamost = tbdata.field("mag3")
hdulist.close()
hdulist = pyfits.open("/home/annaho/aida41040/annaho/TheCannon/examples/example_DR12/allStar-v603.fits")
tbdata = hdulist[1].data
ra_apogee_all = tbdata['RA']
dec_apogee_all = tbdata['DEC']
am_apogee_all = tbdata['PARAM_ALPHA_M']
am_apogee_all = tbdata['AK_WISE']
good_coords = np.logical_and(ra_apogee_all > -90, dec_apogee_all > -90)
good = np.logical_and(good_coords, am_apogee_all > -90)
ra_apogee = ra_apogee_all[good]
dec_apogee = dec_apogee_all[good]
am_apogee = am_apogee_all[good]
hdulist.close()
ra_both = np.hstack((ra_apogee, ra_lamost))
dec_both = np.hstack((dec_apogee, dec_lamost))
am_all = np.hstack((am_apogee, am_lamost))

# create a RA and Dec grid
ra_all = []
dec_all = []
for ra in np.arange(0, 360, 0.5):
    for dec in np.arange(-90, 90, 0.5):
        ra_all.append(ra)
        dec_all.append(dec)

ra = np.array(ra_all)
dec = np.array(dec_all)

# convert RA and Dec to phi and theta coordinates
def toPhiTheta(ra, dec):
    phi = ra * np.pi/180.
    theta = (90.0 - dec) * np.pi / 180.
    return phi, theta

phi, theta = toPhiTheta(ra, dec)
phi_lamost, theta_lamost = toPhiTheta(ra_lamost, dec_lamost)
phi_apogee, theta_apogee = toPhiTheta(ra_apogee, dec_apogee)
phi_all, theta_all = toPhiTheta(ra_both, dec_both)

# to just plot all points, do
#hp.visufunc.projplot(theta, phi, 'bo')
#hp.visufunc.projplot(theta_lamost, phi_lamost, 'bo')
#hp.visufunc.graticule() # just the bare background w/ lines
# more examples are here
# https://healpy.readthedocs.org/en/latest/generated/healpy.visufunc.projplot.html#healpy.visufunc.projplot

## to plot a 2D histogram in the Mollweide projection
# define the HEALPIX level
# NSIDE = 32 # defines the resolution of the map
NSIDE =  128

# find the pixel ID for each point
# pix = hp.pixelfunc.ang2pix(NSIDE, theta, phi)
pix_lamost = hp.pixelfunc.ang2pix(NSIDE, theta_lamost, phi_lamost)
pix_apogee = hp.pixelfunc.ang2pix(NSIDE, theta_apogee, phi_apogee)
pix_all = hp.pixelfunc.ang2pix(NSIDE, theta_all, phi_all)
# pix is in the order of ra and dec

# prepare the map array
m_lamost = hp.ma(np.zeros(hp.nside2npix(NSIDE), dtype='float'))
mask_lamost = np.zeros(hp.nside2npix(NSIDE), dtype='bool')

for pix_val in np.unique(pix_lamost):
    choose = np.where(pix_lamost==pix_val)[0]
    if len(choose) == 1:
        #m_lamost[pix_val] = rmag_lamost[choose[0]]
        m_lamost[pix_val] = am_lamost[choose[0]]
    else:
        #m_lamost[pix_val] = np.median(rmag_lamost[choose])
        m_lamost[pix_val] = np.median(am_lamost[choose])

mask_lamost[np.setdiff1d(np.arange(len(m_lamost)), pix_lamost)] = 1
m_lamost.mask = mask_lamost

m_apogee= hp.ma(np.zeros(hp.nside2npix(NSIDE), dtype='float'))
mask_apogee= np.zeros(hp.nside2npix(NSIDE), dtype='bool')

for pix_val in np.unique(pix_apogee):
    choose = np.where(pix_apogee==pix_val)[0]
    if len(choose) == 1:
        m_apogee[pix_val] = am_apogee[choose[0]]
    else:
        m_apogee[pix_val] = np.median(am_apogee[choose])

mask_apogee[np.setdiff1d(np.arange(len(m_apogee)), pix_apogee)] = 1
m_apogee.mask = mask_apogee

m_all = hp.ma(np.zeros(hp.nside2npix(NSIDE), dtype='float'))
mask_all= np.zeros(hp.nside2npix(NSIDE), dtype='bool')

for pix_val in np.unique(pix_all):
    choose = np.where(pix_all==pix_val)[0]
    if len(choose) == 1:
        m_all[pix_val] = am_all[choose[0]]
    else:
        m_all[pix_val] = np.median(am_all[choose])

mask_all[np.setdiff1d(np.arange(len(m_all)), pix_all)] = 1
m_all.mask = mask_all

# perceptually uniform: inferno, viridis, plasma, magma
cmap=cm.magma
cmap.set_under('w')

# composite map
# plot map ('C' means the input coordinates were in the equatorial system)
rcParams.update({'font.size':16})
#hp.visufunc.mollview(m_apogee, coord=['C','G'], rot=(150, 0, 0), flip='astro',
#        notext=True, title=r'$\alpha$/M for APOGEE DR12', cbar=True,
#        norm=None, min=-0.07, max=0.3, cmap=cmap, unit = r'$\alpha$/M [dex]')
#hp.visufunc.mollview(m_lamost, coord=['C','G'], rot=(150, 0, 0), flip='astro',
#        notext=True, title=r'$\alpha$/M for 500,000 LAMOST giants', cbar=True,
#        norm=None, min=-0.07, max=0.3, cmap=cmap, unit = r'$\alpha$/M [dex]')
        #notext=True, title="r-band magnitude for 500,000 LAMOST giants", cbar=True,
        #norm=None, min=11, max=17, cmap=cmap, unit = r"r-band magnitude [mag]")
hp.visufunc.mollview(m_all, coord=['C','G'], rot=(150, 0, 0), flip='astro',
        notext=True, title=r'$\alpha$/M for APOGEE DR12 + 500,000 LAMOST giants', cbar=True,
        norm=None, min=0.00, max=0.4, cmap=cmap, unit = r'$\alpha$/M [dex]')
hp.visufunc.graticule()

#plt.show()
plt.savefig("full_ak_map.png")
#plt.savefig("apogee_am_map.png")
#plt.savefig("lamost_am_map_magma.png")
#plt.savefig("lamost_rmag_map.png")
