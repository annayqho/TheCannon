#!/usr/bin/python 
#from pyraf import iraf
from numpy import *
from glob import glob
import matplotlib
from matplotlib import pyplot
import pylab
import os, shutil
import numpy
import pyfits
import scipy
import glob
import ephem 
from glob import glob
filein = glob("*allStar2*fits*")[0]
#filein = glob("*all*fits*")[0]
hdulist = pyfits.open(filein)
datain = hdulist[1].data
datain0 = hdulist[0].data
header_table = pyfits.getheader(filein)

#datain - prints all of the header info 
apstarid= datain['APSTAR_ID'] 
fields = datain['FIELD'] 
loc = datain['LOCATION_ID']
pickfield = fields == '4102'
Fehall = datain['METALS']
Fehall_err = datain['METALS_ERR']
loggall = datain['LOGG']
loggall_err = datain['LOGG_ERR']
teffall = datain['TEFF']
teffall_err = datain['TEFF_ERR']
velall = datain['VHELIO_AVG']
velall_err = datain['VERR']
alphaall = datain['ALPHAFE']
J = datain['J']
K = datain['K']
JmK = array(J) - array(K) 
FLAG_1all = datain['PARAMFLAG']
loc = array(loc)
pickit = loc == 4103
appick = apstarid[pickit]
Fehpick = Fehall[pickit]
gpick = loggall[pickit]
Tpick = teffall[pickit]
velpick = velall[pickit]
fehsort = sort(Fehpick)
argis = argsort(Fehpick) 
data = zip( appick[argis], Tpick[argis], gpick[argis], Fehpick[argis]) 
savetxt("4103_data.txt", data, fmt = "%s") 
lval_av = []
bval_av = [] 
loc = array(loc) 
loc_all = []
loc_unique = unique(array(loc) )
lval = array(lval)
bval = array(bval)
for each in loc_unique: 
    pickit = loc == each 
    lval_av.append(median(lval[pickit])) 
    bval_av.append(median(bval[pickit])) 
    loc_all.append(each) 

datafields = zip(loc_all, lval_av, bval_av) 
savetxt("summary.txt", datafields, fmt = "%s") 
        
