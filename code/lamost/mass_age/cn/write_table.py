import numpy as np
from astropy.table import Table, Column
from astropy.io import ascii
import glob

DATA_DIR = "/Users/annaho/Data/Mass_And_Age"

print("writing file")
t = Table()

files = np.array(glob.glob("%s/*all.npz" %DATA_DIR))
for f in files:
    fullname = f.split('/')[-1]
    name = '_'.join(fullname.split('_')[0:-1]) 
    vals = np.array(np.load(f)['arr_0'])
    if name=='id':
        vals = [(val.decode("utf-8")).split('/')[-1] for val in vals]
    t[name] = Column(vals, description=name)

t.write('lamost_catalog.csv', format='ascii.fast_csv')
