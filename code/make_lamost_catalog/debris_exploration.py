import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfits

a = pyfits.open("table_for_paper_full.fits")
b = a[1].data
feh = b['cannon_m_h']
am = b['cannon_alpha_m']
rv_all = b['rv']
rverr = b['rv_err']
rv = rv_all / rverr

good = rverr < 50
feh = feh[good]
am = am[good]
rv = rv[good]

lowlim = -200
highlim = 180
lowlim = -15
highlim = 13

plt.scatter(feh, am, c=rv, s=5, lw=0, vmin=-15, vmax=5, cmap="viridis")
plt.xlim(-2.5, 0.8)
plt.xlabel("[Fe/H]")
plt.ylabel("[alpha/M]")
plt.colorbar(label="RV/RV_err")
plt.ylim(-0.15, 0.55)
plt.show()

fig, axarr=plt.subplots(1,3, sharex=True, sharey=True)

ax = axarr[0]
choose1 = np.logical_and(rv < highlim, rv > lowlim)
ax.scatter(feh[choose1], am[choose1], c='k', alpha=1, s=5, lw=0, label="%s < RV (km/s) < %s" %(str(lowlim), str(highlim)))
ax.set_ylabel("alpha enhancement, a/M (dex)")
ax.legend()

ax = axarr[1]
choose = rv < lowlim 
ax.scatter(feh[choose1], am[choose1], c='k', alpha=1, s=5, 
        lw=0, label="%s < RV (km/s) < %s" %(str(lowlim), str(highlim)))
ax.scatter(feh[choose], am[choose], c='r', alpha=1, s=5, 
        lw=0, label="RV (km/s) < %s" %str(lowlim))
ax.set_xlabel("metallicity, Fe/H (dex")
ax.legend()

ax = axarr[2]
choose = rv > highlim
ax.scatter(feh[choose1], am[choose1], c='k', alpha=1, s=5, lw=0, label="%s < RV (km/s) < %s" %(str(lowlim), str(highlim)))
ax.scatter(feh[choose], am[choose], c='r', alpha=1, s=5, lw = 0, label="RV (km/s) > %s" %str(highlim))
ax.legend()

plt.xlim(-2.5, 0.8)
plt.ylim(-0.15, 0.55)

plt.show()

