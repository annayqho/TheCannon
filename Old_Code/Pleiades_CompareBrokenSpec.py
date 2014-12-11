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

# We make the following spectrum: the average of all the stars that are not broken.
brokenstars = ['aspcapStar-v304-2M03415868+2342263.fits', 'aspcapStar-v304-2M03422154+2439527.fits', 'aspcapStar-v304-2M03453903+2513279.fits', 'aspcapStar-v304-2M03463533+2324422.fits', 'aspcapStar-v304-2M03472083+2505124.fits', 'aspcapStar-v304-2M03473521+2532383.fits', 'aspcapStar-v304-2M03475973+2443528.fits', 'aspcapStar-v304-2M03482277+2358212.fits']

specdata1 = []
specdata2 = []

for file in allfiles:
    filename = "%s/%s" %(base, file)
    a = pyfits.open(filename)
    if file in brokenstars:
        specdata1.append(np.array(a[1].data))
    else:
        specdata2.append(np.array(a[1].data))

specdata1 = np.array(specdata1)
specdata2 = np.array(specdata2)

avgspec = avg(specdata2)

colors = ['r-', 'r--', 'g-', 'g--', 'b-', 'b--', 'm-', 'm--']

plt.plot(avgspec, 'k-', label="All Other Pleiades")

for i in range(0, len(specdata1)):
    dataset = specdata1[i]
    plt.plot(dataset, colors[i], label=brokenstars[i])

plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Spectra of Misbehaving Pleiades')
plt.legend()
plt.show()

