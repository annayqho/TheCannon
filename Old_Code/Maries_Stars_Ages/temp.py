if glob.glob('normed_data.pickle'): 
    file_in2 = open('normed_data.pickle', 'r') 
    dataall, metaall, labels, Ametaall, cluster_name = pickle.load(file_in2)
    file_in2.close()

fn = 'apokasc_all_ages.txt'
T_est,g_est,feh_est,age_est = np.loadtxt(fn, usecols = (6,8,4,10), unpack =1) 
    
labels = ["teff", "logg", "feh", "age" ]
dir = '/home/annaho/AnnaCannon/Code/Maries_Data/'
file_list = []
for file in os.listdir(dir):
    if file.startswith("aspcapStar") and file.endswith(".fits"):
        file_list.append('%s%s' %(dir,file))

a = open(fn, 'r') 
al = a.readlines() 
a.close()
bl = [] # 2MASS ID
for each in al:
    bl.append(each.split()[0])
# currently bl starts with a #
bl.pop(0)
    
for jj,each in enumerate(file_list):
    #print each
    a = pyfits.open(each) 
    b = pyfits.getheader(each) 
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    if jj == 0:
        nmeta = len(labels) # number of parameters
        nlam = len(a[1].data) # number of pixels
    val = diff_wl*(nlam) + start_wl 
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    ydata = (np.atleast_2d(a[1].data))[0] 
    ydata_err = (np.atleast_2d(a[2].data))[0] 
    ydata_flag = (np.atleast_2d(a[3].data))[0] 
    assert len(ydata) == nlam
    wl_full = [10**aval for aval in wl_full_log]
    xdata= np.array(wl_full)
    ydata = np.array(ydata)
    ydata_err = np.array(ydata_err)
    sigma = (np.atleast_2d(a[2].data))[0]# /y1
    if jj == 0:
        npix = len(xdata) # the number of pixels
        dataall = np.zeros((npix, len(file_list), 3))
    if jj > 0:
        assert xdata[0] == dataall[0, 0, 0]

    dataall[:, jj, 0] = xdata
    dataall[:, jj, 1] = ydata
    dataall[:, jj, 2] = sigma

#dataall = continuum_normalize(dataall) #dataall

