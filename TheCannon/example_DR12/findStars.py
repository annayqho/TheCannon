# This script takes in a star's particular 2MASS ID, and finds that 2MASS ID in the allStar-v603.fits file. Then, it uses the index of that ID to figure out the field, and prints the resulting link to the datafile on the SDSS website.

import pyfits

def findIndex(seachFor, searchIn):
    instances = searchIn.count(searchFor)
    if instances > 1: 
        print "something's wrong, star here more than once"
        print searchFor
    elif instances == 0: # if the star isn't there at all...
        print "star isn't here at all"
        return -1 
    return searchIn.index(searchFor)

if __name__ == "__main__": 
    # Open the fits file and read it in

    fitsfile = pyfits.open("allStar-v603.fits")
    fitsIDs = fitsfile[1].data['APSTAR_ID'] # This has 163278 stars
    fitsfields = fitsfile[1].data['LOCATION_ID']    

    # Sample line from fitsIDs: 'apogee.apo1m.s.stars.1.VESTA'

    searchIn = []
    for i in range(0, len(fitsIDs)):
        searchIn.append(fitsIDs[i].split(".")[-1])

    # Now, we have the ID array, which is a list of all of the 2MASS IDs

    # Trial star:
    # 2MASS_ID AlphaFe AlphaFe_err MH FeH FeH_err Teff Teff_err seismic_logg seismic_logg_err max_age
    # J18501318+4139450   0.039  0.05   -0.099  -0.093 0.03  4803   91  2.67    0.01     3.1

    # Goal: put .sh script commands into a .sh file

    outputfile = open('wget_LAMOST_overlap_stars.sh', 'w')

    # Open list of 2MASS IDs for LAMOST xcalib stars
    inputfile = open('2MASSID.txt', 'r')
    inputfile.readline() # ignore the first line
    
    starlist = inputfile.readlines()
    
    for star in starlist:
        searchFor = filter(None, star.split(' '))[0]
        index = findIndex(searchFor, searchIn)
        field = fitsfields[index]
        command = "wget --no-parent http://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/stars/l25_6d/v603/%s/aspcapStar-r5-v603-%s.fits" %(field,searchFor)
        outputfile.write(command + '\n')

    outputfile.close()
    inputfile.close()



    
