def load_labels(filename="example_apokasc/apokasc_all_ages.txt", apogee_ids):
    print("Loading reference labels from file %s" %filename)
    searchIn = np.loadtxt(filename, usecols=(0,), dtype=str)
    labels = np.loadtxt(
            "example_apokasc/apokasc_all_ages.txt", dtype='float', 
            usecols=(1,2,3,4,5,6,7,8,9,10))
    inds = np.array([np.where(searchIn==a)[0][0] for a in apogee_ids]) 
    return labels[inds]
