import os, pyfits, numpy as np

lamost = os.listdir('example_LAMOST/Data/')
apogee = os.listdir('example_DR12/Data/')

print(len(lamost)) #11061 
print(len(apogee)) #11057

ra_lamost = open("ra_lamost.txt", "w")
ra_apogee = open("ra_apogee.txt", "w")
dec_lamost = open("dec_lamost.txt", "w")
dec_apogee = open("dec_apogee.txt", "w")

for star in lamost:
    hdulist = pyfits.open('example_LAMOST/Data/%s' %star)
    ra = str(hdulist[0].header['RA'])
    dec = str(hdulist[0].header['Dec'])
    ra_lamost.write(ra + '\n')
    dec_lamost.write(dec + '\n')

for star in apogee:
    hdulist = pyfits.open('example_DR12/Data/%s' %star)
    ra = str(hdulist[0].header['RA']) 
    dec = str(hdulist[0].header['Dec']) 
    ra_apogee.write(ra + '\n')
    dec_apogee.write(dec + '\n')

ra_lamost.close()
ra_apogee.close()
dec_lamost.close()
dec_apogee.close()
