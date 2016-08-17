import numpy as np

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"
wl = np.load("%s/wl.npz" %DATA_DIR)['arr_0']
npix = len(wl)

# DIBs

all_dibs = [4066, 4180, 4428, 4502, 4726, 4760, 4763, 4780, 4882, 4964, 
        5173, 5404, 5488, 5494, 5508, 5528, 5545, 5705, 5711, 5778, 
        5780, 5797, 5844, 5850,
        6010, 6177, 6196, 6203, 6234, 6270, 6284, 6376, 6379, 6445, 6533,
        6614, 6661, 6699, 6887, 6919, 6993, 7224, 7367, 7562, 8621]

# to my eye, the really prominent ones are
dibs = [4428, 4502, 4726, 4763, 4780, 4882,
        5488, 5494, 5508, 5545, 5705, 5778,
        5780, 5797, 5488, 5850, 6010, 6177, 6196, 
        6203, 6234, 6270, 6284, 6376, 6379, 6445, 
        6533, 6614, 6661, 8621]
dib_left = np.array(
        [4400,4494,4714,4857,5471,5696,5768,5839,6147,6261,6368,8594])

dib_right = np.array(
        [4459,4516,4789,4916,5555,5721,5807,5858,6220,6314,6683,8641])

# Sodium I D
na_left = np.array([5881])
na_right = np.array([5909])

# Tellurics
tel_left = np.array([6380])
tel_right = np.array([8400])

left = np.hstack((dib_left, na_left, tel_left))
right = np.hstack((dib_right, na_right, tel_right))

#left = tel_left
#right = tel_right

mask = np.zeros(npix, dtype=bool)
for ii,val in enumerate(left):
    add = np.logical_and(wl > val, wl < right[ii])
    mask[add] = True

# Cut off the end of the spectrum, because I think there are
# skylines there or something

end = wl > 8750
mask[end] = True

np.savez("mask.npz", mask)
