""" Compile candidates from the individual candidates txt files """
import os
import glob

direc = "/Users/annaho/Data/Li_Giants/All_Candidates"

files = glob.glob(direc + "/*.txt")
outf = open(direc + "/all_candidates.txt")
for f in files:
    candidates = np.loadtxt(f)
    
