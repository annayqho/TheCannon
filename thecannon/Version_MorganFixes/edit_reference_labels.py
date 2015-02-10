inputf = open("reference_labels.txt", "r")
outputf = open("reference_labels_2.txt", "w")

lines = inputf.readlines()
header = filter(None, lines[0].split(' '))

for item in header[0:len(header)-1]:
    outputf.write(item + '\t')
outputf.write(header[-1])

lines = lines[1:]

for line in lines:
    labels = line.split('\t')
    if len(labels) == 17:
        outputf.write(line)
    else:
        trimmed = labels[0:len(labels)-1]
        trimmed[-1] = trimmed[-1] + '\n'
        outputf.write(trimmed[0])
        for i in range(1, len(trimmed)):
            outputf.write('\t' + trimmed[i])

inputf.close()
outputf.close()
