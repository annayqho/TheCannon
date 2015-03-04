inputf = open("reference_labels.txt", "r") # Melissa's file
outputf = open("reference_labels.csv", "w")

inlines = inputf.readlines()

for inline in inlines:
    inline = filter(None, inline.split(" "))
    inline[0] = inline[0].split("/")[-1]
    newline = inline[0]
    while len(inline) >= 10:
        inline.pop(-1)
        inline[-1] = inline[-1] + "\n"
    for i in range(1,len(inline)):
        newline = newline + "," + inline[i]
    outputf.write(newline)

inputf.close()
outputf.close()

# add a header

label_file = "example_LAMOST/reference_labels.csv"
from cannon.helpers import Table
data = Table(label_file) # throws error
