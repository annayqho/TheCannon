# Marie sent me a file with a long list of stars. 
# For each star, I need to find it in the starsin_SFD_Pleiades.txt file

import pyfits
a = pyfits.open("allStar-v304.fits")
b = pyfits.getheader("allStar-v304.fits")

# Print all the headers:
a[1].data:

# One way to query:

datain = a[1].data
loc = datain['LOCATION_ID']
print loc[0] # or whichever you want

# A shortcut:

print a[1].data['LOCATION_ID'][0]

# Now, I want to find a particular star in this file. 

# Take this star: 
# 2MASS_ID AlphaFe AlphaFe_err MH FeH FeH_err Teff Teff_err seismic_logg seismic_logg_err max_age
# J18501318+4139450   0.039  0.05   -0.099  -0.093 0.03  4803   91  2.67    0.01     3.1 





