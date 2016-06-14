# the file allStar_v603_good_vscat_starflag_teff_logg_aspcapflags.fits was
# created using topcat, and contains all of the bad star information except
# for the PARAMFLAG (topcat couldn't read it)
# so, excise the stars with bad PARAMFLAG...

import pyfits
import numpy as np

filein = "allStar_v603_good_vscat_starflag_teff_logg_aspcapflags.fits"
hdulist = pyfits.open(filein)
a = hdulist[1].data
paramflag = a['PARAMFLAG']
teff_flag = paramflag[:,0] != 0
logg_flag = paramflag[:,1] != 0
mh_flag = paramflag[:,3] != 0
alpha_flag = paramflag[:,4] != 0
paramflag_bad = teff_flag | logg_flag | mh_flag | alpha_flag
nstars = len(paramflag_bad)
ids = a['APOGEE_ID']
hdulist.close()

inputf = "allStar_v603_good_vscat_starflag_teff_logg_aspcapflags.txt"
apogee_id_all = np.loadtxt(inputf, usecols=(0,), dtype=str)
labels_all = np.loadtxt(inputf, usecols=(1,2,3,4,5,6,7,8), dtype=float)
apogee_id = apogee_id_all[~paramflag_bad]
labels = labels_all[~paramflag_bad]

np.savez("apogee_dr12_labels", apogee_id, labels)
