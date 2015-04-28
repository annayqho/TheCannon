# make Training_Data.txt and Test_Data.txt lists

import os
import pyfits

# read in everything that's in the Data_All folder
all_objects = np.array(os.listdir("Data_All")) # 11057 overlap objects, already
# excluding the missing_apogee objects
nobjects = len(all_objects)
sizes = np.zeros(nobjects)
start_wl = np.zeros(nobjects)
end_wl = np.zeros(nobjects)

# examine the size of the arrays
for jj,obj in enumerate(all_objects):
    print(jj)
    filein = "Data_All/%s" %obj
    readin = pyfits.open(filein)
    #sizes[jj] = len(readin[0].data[0]) # this is the flux
    start_wl[jj] = (readin[0].data[2])[0]
    end_wl[jj] = (readin[0].data[2])[-1]

# the smallest array is:
jj = 538
obj = all_objects[jj]
filein = "Data_All/%s" %obj
readin = pyfits.open(filein)

# see what the deal is
wl = readin[0].data[2] # wavelength range: 3838.84 -> 9097.04

# the longest array is:
jj = 1 # wavelength range: 3699.99 -> 9099.133

missing_apogee = set(["spec-56201-EG012420S065452V01_sp01-129.fits", "spec-56202-EG021402N263737V01_sp10-087.fits", "spec-56265-EG012217N184057B01_sp07-106.fits", "spec-56344-HD152421S003238B01_sp16-169.fits"])

#test_set = set(all_objects) - missing_apogee
#test_set = np.array(test_set)

outputf = open("Test_Data.txt", "w")

training_set = np.genfromtxt("Training_Data.txt", dtype=str)
training_set = set(training_set)

test_set = set(all_objects) - training_set

for obj in test_set:
    outputf.write(obj + '\n')

outputf.close()


