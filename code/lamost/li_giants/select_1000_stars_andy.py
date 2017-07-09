""" Randomly select 1,000 of the stars that meet Andy's criteria """

from astropy.table import Table
import sys
import numpy as np

datadir = "/Users/annaho/Github_Repositories/TheCannon/data/LAMOST/"

a = Table.read(datadir + "Mass_And_Age/Ho2017_Catalog.fits")
logg = a['logg'] > 3.5
feh = a['MH'] > -0.5
snr = a['SNR'] > 20
chisq = a['Red_Chisq'] < 3
choose = logg & feh & snr & chisq # 55,109
nobj = sum(choose)
randomints = np.random.randint(0,nobj,2000)

ids = a['LAMOST_ID'][choose][randomints]
np.savetxt("random_for_andy_2000.txt", ids, fmt="%s")
