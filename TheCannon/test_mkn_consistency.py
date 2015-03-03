# Fix the data directory and convert into a csv file

inputf = open("test18_badremoved.txt", "r")
outputf = open("reference_labels.csv", "w")

inlines = inputf.readlines()

for inline in inlines:
    inline = filter(None, inline.split(" "))
    inline.pop(1) # remove cluster name
    inline.pop(-1) # remove \n
    inline[0] = inline[0].split("/")[-1]
    inline[-1] = inline[-1]+"\n"
    newline = inline[0]
    for i in range(1,len(inline)):
        newline = newline + "," + inline[i]
    outputf.write(newline)

inputf.close()
outputf.close()

# check label file
label_file = "example_MKN_Check/reference_labels.csv"
from cannon.helpers import Table
data = Table(label_file) # throws error
# Added 0.000 value to the end, or deleted last value, for the following:
# Lines 84-86 have 16 columns instead of 17
# Lines 385-393 have 16 columns instead of 17
# Lines 394-459 have 18 columns instead of 17
# Lines 460-487 have 16 columns instead of 17
# Lines 517-521 have 16 columns instead of 17
data = Table(label_file) # now it works

# identify the two bad stars
# condition 1: logg <= 0.2
# condition 2: teff - teff_corr >= 6000.

# added the header to the file, using the ref label file in example_DR10
