import os, pyfits, numpy as np

ra_apogee = np.loadtxt('ra_apogee.txt', dtype=float)
ra_lamost = np.loadtxt('ra_lamost.txt', dtype=float)
stars_apogee = os.listdir('example_DR12/Data')
stars_lamost = os.listdir('example_LAMOST/Data')
stars_apogee_by_ra = np.array([x for (y,x) in sorted(zip(ra_apogee,stars_apogee))])
stars_lamost_by_ra = np.array([x for (y,x) in sorted(zip(ra_lamost,stars_lamost))])

lamost_sorted = open("lamost_sorted.txt", "w")
apogee_sorted = open("apogee_sorted.txt", "w")

for i in range(len(stars_apogee_by_ra)):
    lamost_sorted.write(stars_lamost_by_ra[i] + '\n')
    apogee_sorted.write(stars_apogee_by_ra[i] + '\n')

lamost_sorted.close()
apogee_sorted.close()
