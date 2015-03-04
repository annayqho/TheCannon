# make a list of flags (0 or 1) corresponding to bad DR12 stars
# as in, stars we wouldn't want to use in the training set

import numpy as np
import pyfits

filein = "allStar-v603.fits"
hdulist = pyfits.open(filein)
datain = hdulist[1].data
aspcapflag = datain["ASPCAPFLAG"]
aspcap_id= datain['ASPCAP_ID']

rot = []
for each in aspcapflag:
    rot.append(each & 10**10)

apid = []
for each in apstarid:
    apid.append(each.split('.')[-1])

rot = np.array(rot)
apid = np.array(apid)

apogee = np.loadtxt("example_DR12/apogee_sorted_by_ra.txt", dtype=str)

# for each element of apogee, find it in apid

inds = []

for element in apogee:
    element = element.split('603-')[1]
    element = element.split('.fits')[0]
    ind = np.where(apid==element)[0][0]
    inds.append(ind)

inds = np.array(inds)

# now find those values of rot

flags = rot[inds]
fileo = open("star_flags.txt", "w")

for flag in flags:
    fileo.write(str(flag)+'\n')

fileo.close()
