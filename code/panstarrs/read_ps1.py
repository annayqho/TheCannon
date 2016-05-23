import pyfits
import numpy as np
import matplotlib.pyplot as plt

ob = pyfits.getdata('apodr12_ps1_allwise.fits')

print("done reading file")

ids = ob['apogee_id']
mag = -2.5 * np.log10(np.clip(ob['median'], 1e-30, np.inf))
err = 2.5/np.log(10)*ob['err']/np.clip(ob['median'], 1e-30, np.inf)
err = np.sqrt((1.3*err)**2.+0.01**2.)

saturation_ps1 = np.asarray([14.0, 14.4, 14.4, 13.8, 13.])
#bad = mag > 50
#bad = mag < saturation_ps1
#starflag = bad.sum(axis=1) > 0

#ids_good = ids[~starflag]
g = mag[:,0]
r = mag[:,1]
i = mag[:,2]
z = mag[:,3]
y = mag[:,4]

good_g = np.logical_and(g>=14.0, g<30)
good_r = np.logical_and(r>=14.4, r<30)
good_i = np.logical_and(i>=14.4, i<30)
good_z = np.logical_and(z>=13.8, z<30)
good_y = np.logical_and(y>=13.0, y<30)

good_gr = np.logical_and(good_g, good_r)
good_iz = np.logical_and(good_i, good_z)
good_griz = np.logical_and(good_gr, good_iz)
good = np.logical_and(good_griz, good_y)

ids = ids[good]
g = g[good]
r = r[good]
i = i[good]
z = z[good]
y = y[good]

err_g = err[:,0][good]
err_r = err[:,1][good]
err_i = err[:,2][good]
err_z = err[:,3][good]
err_y = err[:,4][good]

gi = g-i
ri = r-i
zi = z-i
yi = y-i

err_gi = np.sqrt(err_g**2 + err_i**2)
err_ri = np.sqrt(err_r**2 + err_i**2)
err_zi = np.sqrt(err_z**2 + err_i**2)
err_yi = np.sqrt(err_y**2 + err_i**2)

colors = np.vstack((ids, gi, err_gi, ri, err_ri, zi, err_zi, yi, err_yi)).T

np.savetxt("ps_colors.txt", colors, delimiter=',', header='ids,gi,gi_err,ri,ri_err,zi,zi_err,yi,yi_err', fmt="%s")
