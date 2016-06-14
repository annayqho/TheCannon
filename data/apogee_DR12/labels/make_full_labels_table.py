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
m = datain['PARAM_M_H']
a = datain['PARAM_ALPHA_M']
c = datain['C_H']
n = datain['N_H']
al = datain['AL_H']
ca = datain['CA_H']
fe = datain['FE_H']
k = datain['K_H']
mg = datain['MG_H']
mn = datain['MN_H']
na = datain['NA_H']
ni = datain['NI_H']
o = datain['O_H']
si = datain['SI_H']
s = datain['S_H']
ti = datain['TI_H']
v = datain['V_H']

#vals = datain['FPARAM']
#t = vals[:,0]
#g = vals[:,1]
#m = vals[:,3]
#c = vals[:,4]
#n = vals[:,5]
#a = vals[:,6]


#vscat = datain['VSCATTER']
#starflag = datain['STARFLAGS']
SNR = datain['SNR']
#aspcapflag = datain['ASPCAPFLAG']

# read the real list of test stars
lamost_sorted = np.loadtxt('../lamost_sorted_by_ra.txt', 
        dtype=str)
apogee_sorted = np.loadtxt('../apogee_sorted_by_ra.txt', dtype=str)

# find the indices of the apogee training stars in the apstarid list

inds = []
for star in apogee_sorted:
    ID = star.split('v603-')[1]
    ID = ID.split('.fits')[0]
    inds.append(np.where(apstarid==ID)[0][0])

inds = np.array(inds)

# now find the corresponding training values
ids = apstarid[inds]

snr = SNR[inds]
t = t[inds]
g = g[inds]
m = m[inds]
a = a[inds]
c = c[inds]
n = n[inds]
al = al[inds]
ca = ca[inds]
fe = fe[inds]
k = k[inds]
mg = mg[inds]
mn = mn[inds]
na = na[inds]
ni = ni[inds]
o = o[inds]
si = si[inds]
s = s[inds]
ti = ti[inds]
v = v[inds]

#vals = datain['FPARAM']
#t = vals[:,0]
#g = vals[:,1]
#m = vals[:,3]
#c = vals[:,4]
#n = vals[:,5]
#a = vals[:,6]

#vscatter = vscat[inds]
#starflags = starflag[inds]
#aspcapflags = aspcapflag[inds]
#badbits = 2**23
#flags = np.bitwise_and(aspcapflag, badbits)
#flags[flags!=0] = 1

# and now write the training file

nstars = len(t)
file_out = open("apogee_dr12_labels.csv", "w")
header = 'apogee_id,lamost_id,t,l,m,a,c,n,al,ca,fe,k,mg,mn,na,ni,o,si,s,ti,v,snr\n'

file_out.write(header)

for i in range(nstars):
    print(i)
    line = (str(apogee_sorted[i]) + ',' + str(lamost_sorted[i]) +','+
    str(t[i]) + ',' + str(g[i]) + ',' + str(m[i]) +','+ str(a[i]) + ',' +
    str(c[i]) + ',' + str(n[i]) + ',' + str(al[i]) + ',' + str(ca[i]) + ','+
    str(fe[i]) + ',' + str(k[i]) + ',' + str(mg[i]) + ',' + str(mn[i]) + ','+
    str(na[i]) + ',' + str(ni[i]) + ',' + str(o[i]) + ',' + str(si[i]) + ','+
    str(s[i]) + ',' + str(ti[i]) + ',' + str(v[i]) + ',' + str(snr[i]) + '\n')
    file_out.write(line)

#vals = datain['FPARAM']
#t = vals[:,0]
#g = vals[:,1]
#m = vals[:,3]
#c = vals[:,4]
#n = vals[:,5]
#a = vals[:,6]

#vscatter = vscat[inds]
file_out.flush()
file_out.close()
