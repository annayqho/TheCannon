#!/usr/bin/env python
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

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
phi = ra * np.pi/180.
theta = (90.0 - dec) * np.pi/180.

# to just plot all points, do
hp.visufunc.projplot(theta, phi, 'bo')
hp.visufunc.graticule()
# more examples are here
# https://healpy.readthedocs.org/en/latest/generated/healpy.visufunc.projplot.html#healpy.visufunc.projplot

## to plot a 2D histogram in the Mollweide projection
# define the HEALPIX level
NSIDE = 32

# find the pixel ID for each point
pix = hp.pixelfunc.ang2pix(NSIDE, theta, phi)

# select all points above Dec > -30
pix_unique = np.unique(pix[dec > -30])

# prepare the map array
m = np.zeros(hp.nside2npix(NSIDE), dtype='u2')

# tag the map array with pixels above Dec > -30
m[pix_unique] = 1

# plot map ('C' means the input coordinates were in the equatorial system)
hp.visufunc.mollview(m, coord=['C'], rot=(0, 0, 0), notext=True, title='', cbar=False)
hp.visufunc.graticule()

plt.show()
