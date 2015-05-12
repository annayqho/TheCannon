import pyfits
import os
import numpy as np

hdulist = pyfits.open("allStar-v603.fits")
datain = hdulist[1].data
apstarid= datain['APSTAR_ID']
for i in range(0, len(apstarid)):
    apstarid[i] = apstarid[i].split('.')[-1]

t = datain['TEFF']
g = datain['LOGG']
f = datain['FE_H']
params = datain['PARAM']
a = params[:,-1]

# ignore stars flagged as having unreliable Teff, logg, metallicity, alpha

star_warn = 7
star_bad = 23
a_warn = 4
a_bad = 20
no_result = 31
good_star = (np.bitwise_and(datain['aspcapflag'], 2**star_warn)==0) & \
        (np.bitwise_and(datain['aspcapflag'], 2**star_bad)==0)
good_a = (np.bitwise_and(datain['aspcapflag'], 2**a_warn)==0) & \ 
       (np.bitwise_and(datain['aspcapflag'], 2**a_bad)==0)
good = good_star & good_a & \
       (np.bitwise_and(datain['aspcapflag'], 2**no_result) == 0)


# unreliable Teff is flagged in datain['PARAMFLAG'][0]
# unreliable logg is flagged in datain['PARAMFLAG'][1]
# unreliable metals is flagged in datain['PARAMFLAG'][3]
# unreliable alpha is flagged in datain['PARAMFLAG'[4]
# but in principle, these should also be flagged in datain['ASPCAPFLAG']

good_all = (np.bitwise_and(datain['aspcapflag'],badbits)==0) & (datain['teff_err'] > 0) & (datain['logg_err']>0) & (datain['fe_h_err'] > 0) 

# read the list of all of the overlap stars

lamost_sorted = np.loadtxt('../example_LAMOST/lamost_sorted_by_ra.txt', 
        dtype=str)
apogee_sorted = np.loadtxt('apogee_sorted_by_ra.txt', dtype=str)

# find the apogee equivalents of the ts_lamost stars
# inds...these are the indices, in lamost_sorted, of all the training stars
# and thus these are the indices in apogee_sorted of all the training stars

#inds = []
#for star in ts_lamost:
#    inds.append(np.where(lamost_sorted==star)[0][0])
#inds = np.array(inds)
#ts_apogee = apogee_sorted[inds]

# find the indices of the apogee training stars in the apstarid list

inds = []
for star in apogee_sorted:
    ID = star.split('v603-')[1]
    ID = ID.split('.fits')[0]
    inds.append(np.where(apstarid==ID)[0][0])

inds = np.array(inds)

# now find the corresponding training values...
# this is in the order in which they appear in ts_apogee and ts_lamost

ids = apstarid[inds]
teff = t[inds]
logg = g[inds]
feh = f[inds]
alpha = a[inds]
good = good_all[inds] 

# and now write the training file

nstars = len(teff)

file_out = open("reference_labels.csv", "w")

header = 'id,teff,logg,feh,alpha\n'

file_out.write(header)

for i in range(nstars):
    if good[i] == True:
        line = str(lamost_sorted[i])+','+str(teff[i])+','+str(logg[i])+','+str(feh[i])+','+str(alpha[i])+'\n'
        file_out.write(line)

file_out.flush()
file_out.close()
