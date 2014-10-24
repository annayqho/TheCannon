# Want to add an "ages" and "logAges" column to the starsin_SFD_Pleiades.txt file

# First, store the ages as an array

agefile = open('ages_2.txt', 'r')
ages = agefile.readlines()
agefile.close()

logagefile = open('logAges.txt', 'r')
logages = logagefile.readlines()
logagefile.close()

i = 0

#in_file = open('test14.txt', 'r')
in_file = open('starsin_SFD_Pleiades.txt', 'r')
out_file = open('starsin_SFD_Pleiades_2.txt', 'w')
#out_file = open('test14_ages.txt', 'w')

#headers = in_file.readline()
#out_file.write(headers.strip("\n") + "\t" + "Age" + "\t" + "log(age)" + "\n")

for line in in_file:
    out_file.write(line.strip("\n") + "\t" + ages[i].strip("\n") + "\t" + logages[i])
    i = i+1

in_file.close()
out_file.close()
