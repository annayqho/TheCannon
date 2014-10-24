# We are interested in plotting two averaged Pleiades spectra: one for the group of eight stars that are having trouble in the Cannon procedure, and one for the rest.

import pyfits
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

base = "/home/annaho/AnnaCannon/Code/4259_Pleaides"
allfiles = os.listdir('/home/annaho/AnnaCannon/Code/4259_Pleaides')
allfiles.remove('cal_wget_ac_Pleaides.sh')
allfiles.remove('Problems')

def avg(data):
    sum = data[0]
    
    for i in range(1, len(data)):
        sum = sum + data[i]

    avg = sum/len(data)
    return avg

# Now, we want to take an average of the spectra for all the stars that are broken, and an average of the spectra for all the stars that aren't.

specdata1 = [] # for the broken stars
specdata2 = [] # for the rest

brokenstars = ['aspcapStar-v304-2M03415868+2342263.fits', 'aspcapStar-v304-2M03422154+2439527.fits', 'aspcapStar-v304-2M03453903+2513279.fits', 'aspcapStar-v304-2M03463533+2324422.fits', 'aspcapStar-v304-2M03472083+2505124.fits', 'aspcapStar-v304-2M03473521+2532383.fits', 'aspcapStar-v304-2M03475973+2443528.fits', 'aspcapStar-v304-2M03482277+2358212.fits']

for file in allfiles:
    filename = "%s/%s" %(base, file)
    a = pyfits.open(filename)
    if file in brokenstars:
        specdata1.append(np.array(a[1].data))
    else:
        specdata2.append(np.array(a[1].data))

specdata1 = np.array(specdata1)
specdata2 = np.array(specdata2)

# Now, average all the values in each array

avg1 = avg(specdata1)
avg2 = avg(specdata2)

plt.plot(avg1, 'r-', label='Broken Stars')
plt.plot(avg2, 'b-', label='All Other Stars')
#plt.xlabel('Wavelength (Angstroms)')
#plt.ylabel('Flux')
plt.title('Mean Spectrum for Each Group')
plt.legend()
plt.show()

diff = avg1 - avg2
plt.plot(diff)
plt.show()

