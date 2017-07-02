import pyfits
import numpy as np
from sigfig import round_sig

def ndec(num):
    dec = str(num).split('.')[-1]
    return len(dec)

def fmt_id(id_val):
    split_val = id_val.split('_')
    return split_val[0] + "\_" + split_val[1]

inputf = pyfits.open("/Users/annaho/Github_Repositories/TheCannon/data/LAMOST/Mass_And_Age/Ho2017_Catalog.fits")
dat = inputf[1].data
inputf.close()
choose = dat['in_martig_range']
lamost_id = dat['LAMOST_ID'][choose]
lamost_id = np.array([fmt_id(val) for val in lamost_id])
ra = dat['RA'][choose]
dec = dat['Dec'][choose]
teff = dat['Teff'][choose]
logg = dat['logg'][choose]
mh = dat['MH'][choose]
cm = dat['CM'][choose]
nm = dat['NM'][choose]
am = dat['AM'][choose]
ak = dat['Ak'][choose]
mass = dat['Mass'][choose]
logAge = dat['logAge'][choose]

teff = np.array([int(val) for val in teff])
logg = np.array([round_sig(val,3) for val in logg])
mh = np.array([round_sig(val, 3) for val in mh])
cm = np.array([round_sig(val, 3) for val in cm])
nm = np.array([round_sig(val, 3) for val in nm])
am = np.array([round_sig(val, 3) for val in am])
ak = np.array([round_sig(val, 3) for val in ak])
mass = np.array([round_sig(val, 2) for val in mass])
logAge = np.array([round_sig(val, 2) for val in logAge])

teff_err = dat['Teff_err'][choose]
logg_err = dat['logg_err'][choose]
mh_err = dat['MH_err'][choose]
cm_err = dat['CM_err'][choose]
nm_err = dat['NM_err'][choose]
am_err = dat['AM_err'][choose]
ak_err = dat['Ak_err'][choose]

teff_scat = dat['Teff_scatter'][choose]
logg_scat = dat['logg_scatter'][choose]
mh_scat = dat['MH_scatter'][choose]
cm_scat = dat['CM_scatter'][choose]
nm_scat = dat['NM_scatter'][choose]
am_scat = dat['AM_scatter'][choose]

mass_err = dat['Mass_err'][choose]
logAge_err = dat['logAge_err'][choose]
snr = dat['SNR'][choose]
chisq =dat['Red_Chisq'][choose]

content = '''\\begin{tabular}{cccccccccc} 
\\tableline\\tableline 
LAMOST ID  & RA & Dec & \\teff\ & \logg\ & \mh\ & \cm\ & \\nm\ & \\alpham\ & \\ak\ \\\\
& (deg) & (deg) & (K) & (dex) & (dex) & (dex) & (dex) & (dex) & mag \\\\    
\\tableline
'''

outputf = open("paper_table.txt", "w")
outputf.write(content)

for i in range(0,4):
    outputf.write(
    '%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\ '
    %(lamost_id[i], np.round(ra[i], 5), np.round(dec[i], 5), 
    teff[i], logg[i], mh[i], cm[i], nm[i], am[i], ak[i]))
        #int(teff[i]), round_sig(logg[i], 3), round_sig(mh[i], 3),
    #round_sig(cm[i], 3), round_sig(nm[i], 3), round_sig(am[i], 3), 
    #round_sig(ak[i], 3)))

content = '''\\tableline
\end{tabular}}
\end{table}

\\begin{table}[H]
\\tablenum{1}
\caption{
\\textbf{continued: Formal Errors}}
{\scriptsize
\\begin{tabular}{cccccccc}
\\tableline\\tableline
LAMOST ID  & $\sigma$(\\teff) & $\sigma$(\logg) & $\sigma$(\mh) & $\sigma$(\cm) & $\sigma$(\\nm) & $\sigma$(\\alpham) & $\sigma$(\\ak) \\\\
& (K) & (dex) & (dex) & (dex) & (dex) & (dex) & (mag) \\\\
\\tableline
'''

outputf.write(content)

for i in range(0,4):
    outputf.write(
    '%s & %s & %s & %s & %s & %s & %s & %s \\\\ '
    %(lamost_id[i], int(teff_err[i]), 
    np.round(logg_err[i], ndec(logg[i])),
    np.round(mh_err[i], ndec(mh[i])),
    np.round(cm_err[i], ndec(cm[i])),
    np.round(nm_err[i], ndec(nm[i])),
    np.round(am_err[i], ndec(am[i])),
    np.round(ak_err[i], ndec(ak[i]))))
 

content = '''\\tableline
\end{tabular}}
\end{table}

\\begin{table}[H]
\\tablenum{1}
\caption{
\\textbf{continued: Estimated Error (Scatter)}}
{\scriptsize
\\begin{tabular}{cccccccccc}
\\tableline\\tableline
LAMOST ID  & $s$(\\teff) & $s$(\logg) & $s$(\mh) & $s$(\cm) & $s$(\\nm) & $s$(\\alpham) \\\\
& (K) & (dex) & (dex) & (dex) & (dex) & (dex) \\\\   
\\tableline
'''

outputf.write(content)

for i in range(0,4):
    outputf.write(
    '%s & %s & %s & %s & %s & %s & %s \\\\ '
    %(lamost_id[i], int(teff_scat[i]), 
    np.round(logg_scat[i], ndec(logg[i])),
    np.round(mh_scat[i], ndec(mh[i])),
    np.round(cm_scat[i], ndec(cm[i])),
    np.round(nm_scat[i], ndec(nm[i])),
    np.round(am_scat[i], ndec(am[i]))))

content = '''\\tableline
\end{tabular}}
\end{table}

\\begin{table}[H]
\\tablenum{1}
\caption{
\\textbf{continued}}
{\scriptsize
\\begin{tabular}{ccccccc}
\\tableline\\tableline
LAMOST ID & Mass & log(Age) & $\sigma$(Mass) & $\sigma$(log(Age)) & SNR & Red. \\\\
& ($M_\odot$) & dex & ($M_\odot$) & (dex) & & $\chi^2$ \\\\    
\\tableline
'''

outputf.write(content)

for i in range(0,4):
    outputf.write(
    '%s & %s & %s & %s & %s & %s & %s \\\\ '
    %(lamost_id[i], mass[i], logAge[i],
    np.round(mass_err[i], ndec(mass[i])),
    np.round(logAge_err[i], ndec(logAge[i])),
    round_sig(snr[i], 3), round_sig(chisq[i], 2)))
 
content = '''\\tableline
\end{tabular}}
\end{table}
'''

outputf.write(content)
outputf.close()
