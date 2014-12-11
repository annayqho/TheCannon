# This script takes in a star's particular 2MASS ID, and finds that 2MASS ID in the allStar-v304.fits file. Then, it uses the index of that ID to figure out the field, and prints the resulting link to the datafile on the SDSS website.

# The stars we're interested in, in this case, are the stars from Marie Martig.

import pyfits

def findIndex(seachFor, searchIn):
    instances = searchIn.count(searchFor)
    if instances > 1: print "something's wrong, star here more than once"
    elif instances == 0: # if the star isn't there at all...
        print "star isn't here at all"
        return -1 
    return searchIn.index(searchFor)

if __name__ == "__main__": 
    # Open the fits file and read it in

    fitsfile = pyfits.open("allStar-v304.fits")
    fitsIDs = fitsfile[1].data['APSTAR_ID'] # This has 59607 stars
    fitsfields = fitsfile[1].data['LOCATION_ID']    

    # Sample line from fitslist: 'apogee.n.s.s3.4264.2M00000032+5737103'

    searchIn = []
    for i in range(0, len(fitsIDs)):
        searchIn.append(fitsIDs[i].split("2M")[-1])

    # Now, we have the ID array, which is a list of all of the 2MASS IDs

    # Trial star:
    # 2MASS_ID AlphaFe AlphaFe_err MH FeH FeH_err Teff Teff_err seismic_logg seismic_logg_err max_age
    # J18501318+4139450   0.039  0.05   -0.099  -0.093 0.03  4803   91  2.67    0.01     3.1

    # Goal: put .sh script commands into a .sh file

    outputfile = open('wget_Maries_stars.sh', 'w')

    # Open Marie's data file
    inputfile = open('apokasc_all_ages.txt', 'r')
    inputfile.readline() # ignore the first line
    
    starlist = inputfile.readlines()
    
    for star in starlist:
        searchFor = star.split(' ')[0][1:] # just the number
        index = findIndex(searchFor, searchIn)
        field = fitsfields[index]
        command = "wget --no-parent http://data.sdss3.org/sas/dr10/apogee/spectro/redux/r3/s3/a3/v304/%s/aspcapStar-v304-2M%s.fits" %(field,searchFor)
        outputfile.write(command + '\n')

    outputfile.close()
    inputfile.close()



    
