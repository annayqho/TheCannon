import os
import pyfits
import numpy as np

training_set_size = 1200

# read in all files from LAMOST data directory: these are the "working stars"

stars_lamost = np.array(os.listdir('example_LAMOST/Data'))
nstars = len(stars_lamost)

# find corresponding SNR, BAD_STAR, APOGEE values

# the following should all be in the same consistent order...
snrs = np.loadtxt('example_LAMOST/snr_list.txt', dtype=float)
flags = np.loadtxt('example_DR12/star_flags.txt', dtype=float)
sorted_lamost = np.loadtxt('example_LAMOST/lamost_sorted_by_ra.txt', dtype=str)
sorted_apogee = np.loadtxt('example_DR12/apogee_sorted_by_ra.txt', dtype=str)

# for each star in stars_lamost, get its value from these other arrays 
inds = []

for star in stars_lamost:
    inds.append(np.where(sorted_lamost==star)[0][0])

inds = np.array(inds)

stars_snrs = snrs[inds] 
stars_flags = flags[inds]
stars_apogee = sorted_apogee[inds]

# assign all bad stars a snr of 0

bad = stars_flags != 0.
stars_snrs[bad] = 0

# sort working lamost stars by SNR

lamost_sorted_by_snr = np.array([x for (y,x) in sorted(zip(stars_snrs,stars_lamost))])
apogee_sorted_by_snr = np.array([x for (y,x) in sorted(zip(stars_snrs,stars_apogee))])
ts_lamost = lamost_sorted_by_snr[nstars-training_set_size:]
ts_apogee = apogee_sorted_by_snr[nstars-training_set_size:]

# move LAMOST training spec to folder

os.system("mkdir example_LAMOST/Training_Data")
for lamost_file in ts_lamost:
    os.system("cp example_LAMOST/Data/%s example_LAMOST/Training_Data" %lamost_file)

# make the reference label file

#file_in = open("example_LAMOST/reference_labels.csv", 'r')
#file_out = open("example_LAMOST/reference_labels_update.csv", 'w')

#lines = file_in.readlines()
#lines = lines[1:]

#filenames = []
#for line in lines:
#    filename = line.split(',')[0]
#    filenames.append(filename)
#    if filename in ts_apogee:
#        file_out.write(line)

#file_in.close()
#file_out.close()
