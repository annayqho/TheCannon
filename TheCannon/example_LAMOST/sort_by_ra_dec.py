import os, numpy as np

# read in the RA & Dec lists

ra_apogee = np.loadtxt('ra_apogee.txt', dtype=float)
dec_apogee = np.loadtxt('dec_apogee.txt', dtype=float)
ra_lamost = np.loadtxt('ra_lamost.txt', dtype=float)
dec_lamost = np.loadtxt('dec_lamost.txt', dtype=float)

# compare the RA distributions...

ra_lamost = np.round(ra_lamost, 4)
ra_apogee = np.round(ra_apogee, 4)
ra_lamost_sorted = np.array(sort(ra_lamost))
ra_apogee_sorted = np.array(sort(ra_apogee))

# maybe plot RA and Dec and see if there's overlap? 

# read in the file names

stars_apogee = os.listdir('example_DR12/Data')
stars_lamost = os.listdir('example_LAMOST/Data')

# sort each list of file names by RA

stars_apogee_by_ra = np.array([x for (y,x) in sorted(zip(ra_apogee,stars_apogee))])
stars_lamost_by_ra = np.array([x for (y,x) in sorted(zip(ra_lamost,stars_lamost))])

# find the four stars

# for each one:
ra_lamost_sorted = np.delete(ra_lamost_sorted, 298)
print(stars_lamost_by_ra[298])
# "spec-56201-EG012420S065452V01_sp01-129.fits"
stars_lamost_by_ra = np.delete(stars_lamost_by_ra, 298)

ra_lamost_sorted = np.delete(ra_lamost_sorted, 300)
print(stars_lamost_by_ra[300])
# spec-56265-EG012217N184057B01_sp07-106.fits
stars_lamost_by_ra = np.delete(stars_lamost_by_ra, 300)

ra_lamost_sorted = np.delete(ra_lamost_sorted, 336)
print(stars_lamost_by_ra[336]) 
# spec-56202-EG021402N263737V01_sp10-087.fits
stars_lamost_by_ra = np.delete(stars_lamost_by_ra, 336)

ra_lamost_sorted = np.delete(ra_lamost_sorted, 7820)
print(stars_lamost_by_ra[7820])
# spec-56202-EG021402N263737V01_sp10-087.fits
stars_lamost_by_ra = np.delete(stars_lamost_by_ra, 7820)


scatter(ra_lamost_sorted[0:11000], ra_apogee_sorted[0:11000])
plot(ra_lamost_sorted[0:11000]-ra_apogee_sorted[0:11000])
"spec-56208-EG020224N124106L_sp09-091.fits"

