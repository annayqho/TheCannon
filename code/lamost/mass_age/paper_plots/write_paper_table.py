import pyfits
import numpy as np

inputf = pyfits.open("/Users/annaho/Data/LAMOST/Mass_And_Age/Ho2016b_Catalog.fits")
dat = inputf[1].data
inputf.close()
lamost_id = dat['LAMOST_ID']
ra = dat['RA']
dec = dat['Dec']
teff = dat['Teff']
logg = dat['logg']
mh = dat['MH']
cm = dat['CM']
nm = dat['NM']
am = dat['AM']
ak = dat['Ak']
teff_err = dat['Teff_err']
logg_err = dat['logg_err']
mh_err = dat['MH_err']
cm_err = dat['CM_err']
nm_err = dat['NM_err']
am_err = dat['AM_err']
ak_err = dat['Ak_err']

content = '''\\begin{tabular}{cccccccccc} 
\\tableline\\tableline 
LAMOST ID  & RA & Dec & \\teff\ & \logg\ & \mh\ & \cm\ & \\nm\ & \\alpham\ & \\ak\ \\ 
& (deg) & (deg) & (K) & (dex) & (dex) & (dex) & (dex) & (dex) & mag \\    
\\tableline
'''

outputf = open("paper_table.txt", "w")
outputf.write(content)

for i in range(0,4):
    outputf.write(
    '%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\'
    %(lamost_id[i], ra[i], dec[i], teff[i], logg[i], mh[i], cm[i], nm[i], am[i], ak[i]))



# content = '''\\tableline
# \end{tabular}}
# \end{table}
# 
# \\begin{table}[H]
# \caption{
# Continued from Table 1: Formal Errors}
# {\scriptsize
# \\begin{tabular}{cccccccc}
# \\tableline\\tableline
# LAMOST ID  & $\sigma$(\\teff) & $\sigma$(\logg) & $\sigma$(\mh) & $\sigma$(\cm) & $\sigma$(\\nm) & $\sigma$(\\alpham) & $\sigma$(\\ak) \\
# & (K) & (dex) & (dex) & (dex) & (dex) & (dex) & (mag) \\
# \\tableline
# spec-55859-F5902\_sp01... & 3290 & 0.010 & 0.0027 & 0.0043 & 0.0089 & 
# 0.00066 & 0.00036  \\
# spec-55859-F5902\_sp03... & 73.8 &  0.00031 & 8.0e-5 & 0.00012 & 0.0003 & 2.67e-5 & 5.2e-5 \\
# spec-55859-F5902\_sp06... & 65.0 & 0.0004 & 0.0001 & 7.9e-5 & 0.00014 & 
# 4.93e-5 & 7.0e-5 \\
# spec-55859-F5902\_sp08... & 5150 & 0.016 & 0.0047 & 0.0042 & 0.0058 & 
# 0.0015 & 0.00041 \\
# \\tableline
# \end{tabular}}
# \end{table}
# 
# \\begin{table}[H]
# \caption{
# Continued from Table 2: Estimated Error (Scatter)}
# {\scriptsize
# \\begin{tabular}{cccccccccc}
# \\tableline\\tableline
# LAMOST ID  & $s$(\\teff) & $s$(\logg) & $s$(\mh) & $s$(\cm) & $s$(\\nm) & $s$(\\alpham) \\
# & (K) & (dex) & (dex) & (dex) & (dex) & (dex) \\   
# \\tableline
# spec-55859-F5902\_sp01-034 & 78 & 0.14 & 0.078 & 0.085 & 0.20 & 0.043  \\
# spec-55859-F5902\_sp03-209 & 46 &  0.16 & 0.078 & 0.084 & 0.48 & 0.056 \\
# spec-55859-F5902\_sp06-160 & 44 & 0.12 & 0.053 & 0.066 & 0.38 & 0.037 \\
# spec-55859-F5902\_sp08-146 & 88 & 0.16 & 0.092 & 0.095 & 0.18 & 0.050 \\
# \\tableline
# \end{tabular}}
# \end{table}
# 
# \\begin{table}[H]
# \caption{
# Continued from Table 3}
# {\scriptsize
# \\begin{tabular}{ccccccc}
# \\tableline\\tableline
# LAMOST ID & Mass & log(Age) & $\sigma$(Mass) & $\sigma$(log(Age)) & SNR & Red. \\
# & ($M_\odot$) & dex & ($M_\odot$) & (dex) & & $\chi^2$ \\    
# \\tableline
# spec-55859-F5902\_sp01-034 &  0.78 & 1.0 & 0.33 & 0.34 & 33.7 & 0.44 \\
# spec-55859-F5902\_sp03-209 &  1.0 & 0.85 & 0.097 & 0.12 & 169 & 1.7 \\
# spec-55859-F5902\_sp06-160 & 0.43 & 0.34 & 1.3 & 0.66 & 130 & 1.2 \\
# spec-55859-F5902\_sp08-146 & 0.47 & 0.45 & 1.2 & 0.65 & 19.9 & 0.51 \\
# \\tableline
# \end{tabular}}
# \end{table}
# '''

outputf.close()
