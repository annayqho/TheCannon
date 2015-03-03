import numpy as np

file_in = open("reference_labels.csv", 'r')
file_out = open("reference_labels_testing.csv", 'w')

lines = file_in.readlines()
lines = lines[1:]

for line in lines:
    filename = line.split(',')[0]
