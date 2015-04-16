import os

ts_lamost = os.listdir("../example_LAMOST/Training_Data")

current_labels_file = "reference_labels.csv"
current_labels = np.loadtxt("reference_labels.csv", dtype=str)

new_labels_file = "reference_labels_new.csv"
outputf = open(new_labels_file, "w")
header = 'id,teff,logg,feh\n'
file_out.write(header)

# for each row in the current_labels file, if the star is in
# ts_lamost, write that row to the new file

for row in current_labels:
    star = row.split(',')[0]
    if star in ts_lamost:
        file_out.write(star)

outputf.close()
