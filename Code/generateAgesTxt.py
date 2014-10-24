# This script is for generating the ages.txt file

import math as m

numclusters = 19

# GC and OC cluster names
names = ["M92", "M15", "M53", "N5466", "N4147", "M13", "M2", "M3", "M5", "M107", "M71", "N2158", "N2420", "Pleiades", "N7789", "M67", "N6819", "N188", "N6791"] 

# The number of files corresponding to each cluster
filecount = [48, 11, 16, 8, 3, 71, 18, 73, 103, 18, 6, 10, 9, 74, 5, 23, 29, 5, 23]

# The age of each cluster
ages = [12.75, 12.75, 12.25, 12.50, 12.25, 12.00, 11.75, 11.75, 11.50, 12.00, 11.00, 1.054, 1.117, .1352, 1.718, 2.564, 1.493, 4.285, 4.395]

# log(age)
logAges = [10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.0, 9.023, 9.048, 8.131, 9.235, 9.409, 9.174, 9.632, 9.643]

# Print age info

for i in range(0, numclusters):
    print names[i]
    print ages[i]
    print logAges[i]

# Generate a txt file with the raw ages in Gyr

#newfile = open("ages_2.txt", 'w')

#for i in range(0, numclusters):
#    for j in range(0, filecount[i]):
#    	newfile.write(str(ages[i])+"\n")

#newfile.close()

# Generate a txt file with the log(ages)

#newfile = open("logAges.txt", 'w')

#for i in range(0, numclusters):
#    for j in range(0, filecount[i]):
#	newfile.write(str(logAges[i])+"\n")

#newfile.close()








