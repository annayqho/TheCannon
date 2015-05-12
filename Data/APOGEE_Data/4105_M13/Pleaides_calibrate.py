#/usr/python/bin 
import pyfits 

fn = "starsin_new_all_ordered.txt"
fn = "starsin_new_all_ordered.txt"
T_est,g_est,feh_est = np.loadtxt(fn, usecols = (4,6,8), unpack =1) 
a = open(fn, 'r') 
al = a.readlines() 
bl = []
name = [] 
id_star = [] 
for each in al:
  bl.append(each.split()) 
for each in bl:
    name.append(each[1]) 
    id_star.append( each[0].split('-v304-')[1].split('.fits')[0]) 
unames = unique(name) 
ind_name = [] 
unames_ordered  = [] 
len_each =[] 
name = array(name) 
start_u = []
stop_u =[] 
id_u = [] 
id_star = array(id_star)
name = array(name) 
#for each in unames:
##    ind1 = name == each  
#    len_each.append(len(ind1[ind1]) ) 
#    start_u.append(list(name).index(each))
#    id_u.append(id_star[each]) 

file1 = 'stars_all.list' 
dir1 = '/Users/ness/Downloads/Apogee_raw/calibration_apogeecontinuum2/code/4259_Pleaides/'
a = open(dir1+file1, 'r')
al = a.readlines()
bl = []
J = [] 
K = [] 
H = [] 
obj = []
ra = []
dec = [] 
ebmv = [] 
akwise = [] 
for each in al:
    bl.append(each.strip()) 
for each in bl:
    datain = pyfits.open(dir1+each) 
    datain1 = datain[1].data
    datain0 = datain[0].header
    obj.append(datain0["OBJ"]) 
    J.append(datain0["J"]) 
    H.append(datain0["H"]) 
    K.append(datain0["K"]) 
    ra.append(datain0["RA"])
    dec.append(datain0["DEC"])
    ebmv.append(datain0["SFD_EBV"])
    akwise.append(datain0["AKWISE"])
ebmv2 = genfromtxt("extinction_Pleaides_edit.txt", usecols = (3,), unpack =1) 
J,H,K = genfromtxt("JmK_edit_Pleaides.txt", usecols = (11,15,19), unpack =1) 
ebmv = array(ebmv) 
ejmk2 = 0.535*ebmv2
JmK = array(J) - array(K) 
JmKo = JmK  - ejmk2 
JmKo  = JmK - 0.03*0.535
feh = 0.03
JmKo  = JmK - 0.03*0.535
JmKo = JmK  - array(akwise)/1.62 
#JmKo = JmK  - ejmk2 

def theta(b0,b1,b2,b3,b4,b5, feh, JmKo):
    result = b0+b1*JmKo + b2*JmKo*JmKo + b3*JmKo*feh + b4*feh + b5*feh*feh
    return 5040.0/result
b0,b1,b2,b3,b4,b5,Na,Nb = 0.6524,0.5813,0.1225, -0.0646, 0.0370,0.0016,436, 139
#giants = b0,b1,b2,b3,b4,b5,b6 = 0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.013


# now put on isochrone below 
iso = './isochrones_Pad/Pleaides_0160.dat' 
iso = './isochrones_Pad/125Myr_0.0155.txt' 
#iso = 'M35_0140.dat' 
logt, g_iso  = genfromtxt(iso, usecols = (5,6), unpack =1) 
t_iso = 10**logt
file2 = '/Users/ness/Downloads/Apogee_raw/calibration_apogeecontinuum2/code/starsin_new_all_ordered.txt'
t,g,feh = loadtxt(file2, usecols = (3,5,7), unpack = 1) 
a = open(file2)
al = a.readlines()
bl = []
name = [] 
id_star = [] 
for each in al:
  bl.append(each.split()) 
for each in bl:
    name.append(each[1]) 
    id_star.append( each[0].split('-v304-')[1].split('.fits')[0]) 

name = array(name) 
sel_cluster = name == 'Pleiades' 
#sel_cluster = name == 'M35' 

gpick = g[sel_cluster]
tpick = t[sel_cluster] 
fehpick = feh[sel_cluster]
#temperatures = theta(b0,b1,b2,b3,b4,b5,fehpick , JmKo) 
temperatures = theta(b0,b1,b2,b3,b4,b5,0.03 , JmKo) 
fehpick = [0.03]*len(temperatures) 
tpick_IR = temperatures
y_new_all = [] 


for each in tpick_IR: 
    sel_t = logical_and(t_iso > each -400, t_iso < each + 400 ) 
    sel = logical_and(g_iso > 4, sel_t) 
    t_pick = t_iso[sel]
    g_pick = g_iso[sel]
    t_new = arange(min(t_pick), max(t_pick), 1) 
    g_new = arange(min(g_pick), max(g_pick), 0.01) 
    fa = interpolate.interp1d(t_pick, g_pick)
    new_ydata = fa(each) 
    y_new_all.append(new_ydata) 

#    z = np.polyfit(t_pick, g_pick, 1)
#    f = np.poly1d(z)
#    y_new = f(each) 
#    plot(each, y_new, 'r' ) 
#    y_new_all.append(y_new) 

t_new = tpick_IR
g_new = [round(a, 2) for a in y_new_all ] 
feh_new = [a+0.00 for a in fehpick] 
akwise = array(akwise) 
pick = akwise < 10.15
t_new = array(t_new)
g_new= array(g_new)
feh_new = array(feh_new ) 
newparams = zip(t_new[pick], g_new[pick], feh_new[pick]) 
savetxt("Pleaides_new_params_g_feh.txt", newparams, fmt = "%5.5s %5.5s %5.5s" ) 
a = open('Pleaides.txt', 'r') 
al  = a.readlines()
bl = []
for each in al:
    bl.append(each.strip('\n') ) 
bl = array(bl) 
cl = bl[pick] 
#savetxt("Pleaides_cut.txt", cl, fmt = "%s" ) 
