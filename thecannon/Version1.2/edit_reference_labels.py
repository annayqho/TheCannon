inputf = open("reference_labels.txt", "r")
outputf = open("reference_labels_update.txt", "w")

lines = inputf.readlines()
outputf.write(lines[0])
lines = lines[1:]

for line in lines:
    newline = line.split('/')[2]
    outputf.write(newline)

inputf.close()
outputf.close()
