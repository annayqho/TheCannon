"""
This file is part of The Cannon analysis project.
Copyright 2014 Melissa Ness.

# urls
- http://iopscience.iop.org/1538-3881/146/5/133/suppdata/aj485195t4_mrt.txt for calibration stars 
- http://data.sdss3.org/irSpectrumDetail?locid=4330&commiss=0&apogeeid=2M17411636-2903150&show_aspcap=True object explorer 
- http://data.sdss3.org/basicIRSpectra/searchStarA
- http://data.sdss3.org/sas/dr10/apogee/spectro/redux/r3/s3/a3/ for the data files 

# to-do
- need to add a test that the wavelength range is the same - and if it isn't interpolate to the same range 
- format PEP8-ish (four-space tabs, for example)
- take logg_cut as an input
- extend to perform quadratic fitting
"""

import pyfits
import scipy 
import glob 
import pickle
import pylab 
from scipy import interpolate 
from scipy import ndimage 
from scipy import optimize as opt
import numpy as np
LARGE = 1e2 # sigma value to use for bad continuum-normalized data; MAGIC

# This is a goodness_fit calculated for every star
def get_goodness_fit(fn_pickle, filein, Params_all):
    fd = open(fn_pickle,'r')
    dataall, metaall, labels, offsets, coeffs, covs, scatters, chis, chisq = pickle.load(fd)
    fd.close()
    file_with_star_data = str(filein)+".pickle"
    file_with_star_tags = str(filein)+"_tags2.pickle"
    f_flux = open(file_with_star_data, 'r')
    if filein != 'normed_data':
        flux = pickle.load(f_flux)
    if filein == 'normed_data':
        flux, metaall, labels, Ametaall, cluster_name = pickle.load(f_flux)
    f_flux.close()
    labels = Params_all
    nlabels = np.shape(labels)[1]
    nstars = np.shape(labels)[0]
    features_data = np.ones((nstars, 1))
    offsets = np.mean(labels, axis = 0)
    features_data = np.hstack((features_data, labels - offsets))
    newfeatures_data = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (labels - offsets)])
    features_data = np.hstack((features_data, newfeatures_data))
    chi2_all = np.zeros(nstars)
    for jj in range(nstars):
        model_gen = np.dot(coeffs,features_data.T[:,jj])
        data_star = flux[:,jj,1]
        Cinv = 1. / (flux[:,jj, 2] ** 2 + scatters ** 2) # invvar slice of data
        #chi = np.sqrt(Cinv) * (data_star - np.dot(coeffs, features_data.T[:,jj]))
        # The below line without "sum" would give you the chi2 at every wavelength
        chi2 = sum( (Cinv) * (data_star - np.dot(coeffs, features_data.T[:,jj]))**2)
        #chi2 = (Cinv)*(model_gen - data_star)**2
        chi2_all[jj] = chi2

    #chi2_def = chi2/len(xdata)*1
    chi2_def = chi2_all/8575
    chi2_avg = sum(chi2_def)/len(chi2_def)
    chi2_stdev = np.std(chi2_def)
    print "Average Chi2: %s" %str(chi2_avg)
    print "Stdev: %s" %str(chi2_stdev)
    #return chi2_all # probably return it without the sum, and then when you call it save those values to a .pickle file: pickle.dump((Params_all, covs_all,chi2_def,ids), file_in) -- being saved into self_2nd_order_tags.pickle

def weighted_median(values, weights, quantile):
    """weighted_median

    keywords
    --------

    values: ndarray
        input values

    weights: ndarray
        weights to apply to each value in values

    quantile: float
        quantile selection

    returns
    -------
    val: float
        median value
    """
    sindx = np.argsort(values)
    cvalues = 1. * np.cumsum(weights[sindx])
    cvalues = cvalues / cvalues[-1]
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    indx = foo[0]
    return values[indx]

def continuum_normalize(dataall, delta_lambda=50):
    """continuum_normalize

    keywords
    --------

    dataall: ndarray, shape=(Nlambda, Nstar, 3)
        wavelengths, flux densities, errors

    delta_lambda:
        half-width of median region in angstroms


    returns
    -------
    continuum:     (Nlambda, Nstar)
        continuum level

    .. note::

        * does a lot of stuff *other* than continuum normalization

    .. todo::

        * bugs: for loops!
    """
    Nlambda, Nstar, foo = dataall.shape
    continuum = np.zeros((Nlambda, Nstar))
    # sanitize inputs
    for jj in range(Nstar):
        bad_a = np.logical_or(np.isnan(dataall[:, jj, 1]) ,np.isinf(dataall[:,jj, 1]))
        bad_b = np.logical_or(dataall[:, jj, 2] <= 0. , np.isnan(dataall[:, jj, 2]))
        bad = np.logical_or( np.logical_or(bad_a, bad_b) , np.isinf(dataall[:, jj, 2]))
        dataall[bad, jj, 1] = 0.
        dataall[bad, jj, 2] = np.Inf #LARGE#np.Inf #100. #np.Inf
        continuum = np.zeros((Nlambda, Nstar))
    assert foo == 3
    for star in range(Nstar):
        print "get_continuum(): working on star" ,star
        for ll, lam in enumerate(dataall[:, 0, 0]):
            if dataall[ll, star, 0] != lam:
                print dataall[ll,star,0], lam , dataall[ll,0,0] 
                print ll, star 
                print ll+1, star+1, dataall[ll+1, star+1, 0], dataall[ll+1,0,0] 
                print ll+2, star+2, dataall[ll+2, star+2, 0], dataall[ll+2,0,0] 
                assert False
            indx = (np.where(abs(dataall[:, star, 0] - lam) < delta_lambda))[0]
            ivar = 1. / (dataall[indx, star, 2] ** 2)
            ivar = np.array(ivar)
            continuum[ll, star] = weighted_median(dataall[indx, star, 1], ivar, 0.90)
    for jj in range(Nstar):
        bad = np.where(continuum[:,jj] <= 0) 
        continuum[bad,jj] = 1.
        dataall[:, jj, 1] /= continuum[:,jj]
        dataall[:, jj, 2] /= continuum[:,jj]
        dataall[bad,jj, 1] = 1. 
        dataall[bad,jj, 2] = LARGE 
        bad = np.where(dataall[:, jj, 2] > LARGE) 
        dataall[bad,jj, 1] = 1. 
        dataall[bad,jj, 2] = LARGE 
    return dataall 

def get_normalized_test_data(testfile): 
  """
  inputs
    ------
    testfile: str
        the file in with the list of fits files want to test - if normed, move on,
        if not normed, norm it
    also save a SNR file with list of SNR values if it is not present 

    returns
    -------
    testdata:

  """
  name = testfile.split('/')[-2]
  testdir = testfile.split('stars')[0]
  if glob.glob(name+'.pickle'):
      file_in2 = open(name+'.pickle', 'r') 
      testdata = pickle.load(file_in2)
      file_in2.close()
      return testdata 

  a = open(testfile, 'r')
  al2 = a.readlines()
  bl2 = []
  for each in al2:
    bl2.append(testdir+each.strip())
  for jj,each in enumerate(bl2):
    a = pyfits.open(each) 
    if shape(a[6].data) != (8575,):
      ydata = a[1].data[0]
      ysigma = a[2].data[0]
      len_data  = len(a[1].data[0])
      if jj == 0:
          nlam = len(a[1].data[0])
          testdata = np.zeros((nlam, len(bl2), 3))
    if shape(a[1].data) == (8575,):
      ydata = a[1].data
      ysigma = a[2].data
      len_data  = len(a[1].data)
      if jj == 0:
          nlam = len(a[1].data)
          testdata = np.zeros((nlam, len(bl2), 3))
    
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    if jj == 0:
        nlam = len(a[1].data)
        testdata = np.zeros((nlam, len(bl2), 3))
    val = diff_wl*(nlam) + start_wl 
    wl_full_log = np.arange(start_wl,val, diff_wl) 
    wl_full = [10**aval for aval in wl_full_log]
    xdata = wl_full
    testdata[:, jj, 0] = xdata
    testdata[:, jj, 1] = ydata
    testdata[:, jj, 2] = ysigma
  testdata = continuum_normalize(testdata) # testdata
  file_in = open(name+'.pickle', 'w')  
  pickle.dump(testdata,  file_in)
  file_in.close()
  return testdata 

def get_normalized_training_data():
  """
    inputs
    ------
    testfile: str
        the file in with the list of fits files  in the training data
        if not normed, norm it
    have brittle iputs of the test14 which is the training data labels from ASPCAP and the ages which is the ages made by me.Note the Pleiades is my trainng labels for Teff, log g 

    returns
    -------
    dataall - the normalised test data 

  """
  if glob.glob('normed_data.pickle'): 
        file_in2 = open('normed_data.pickle', 'r') 
        dataall, metaall, labels, Ametaall, cluster_name = pickle.load(file_in2)
        file_in2.close()
        return dataall, metaall, labels, Ametaall, cluster_name
  fn = 'starsin_SFD_Pleiades.txt'
  T_est,g_est,feh_est = np.loadtxt(fn, usecols = (4,6,8), unpack =1) 
  T_A,g_A,feh_A = np.loadtxt(fn, usecols = (3,5,7), unpack =1) # these are APOGEE values - interesting to compare e.g for Pleiades and if I get my own temperatures 
  age_est = np.loadtxt('logAges.txt', usecols = (0,), unpack =1) 
  labels = ["teff", "logg", "feh", "age" ]
  a = open(fn, 'r') 
  al = a.readlines() 
  bl = []
  cluster_name = [] 
  for each in al:
    bl.append(each.split()[0]) 
    cluster_name.append(each.split()[1]) 
  for jj,each in enumerate(bl):
    each = each.strip('\n')
    a = pyfits.open(each) 
    b = pyfits.getheader(each) 
    start_wl =  a[1].header['CRVAL1']
    diff_wl = a[1].header['CDELT1']
    print np.atleast_2d(a[1].data).shape
    if jj == 0:
      nmeta = len(labels)
      nlam = len(a[1].data)
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
    starname2 = each.split('.fits')[0]+'.txt'
    sigma = (np.atleast_2d(a[2].data))[0]# /y1
    if jj == 0:
      npix = len(xdata) 
      dataall = np.zeros((npix, len(bl), 3))
      metaall = np.ones((len(bl), nmeta))
      Ametaall = np.ones((len(bl), nmeta))
    if jj > 0:
      assert xdata[0] == dataall[0, 0, 0]

    dataall[:, jj, 0] = xdata
    dataall[:, jj, 1] = ydata
    dataall[:, jj, 2] = sigma

    for k in range(0,len(bl)): 
      # must be synchronised with labels 
      metaall[k,0] = T_est[k] 
      metaall[k,1] = g_est[k] 
      metaall[k,2] = feh_est[k] 
      metaall[k,3] = age_est[k] 
      Ametaall[k,0] = T_A[k]
      Ametaall[k,1] = g_A[k]
      Ametaall[k,2] = feh_A[k]
  dataall = continuum_normalize(dataall) #dataall

  file_in = open('normed_data.pickle', 'w')  
  pickle.dump((dataall, metaall, labels, Ametaall, cluster_name),  file_in)
  file_in.close()
  return dataall, metaall, labels, Ametaall, cluster_name

def do_one_regression_at_fixed_scatter(data, features, scatter):
    """
    Parameters
    ----------
    data: ndarray, [nobjs, 3]
        wavelengths, fluxes, invvars

    meta: ndarray, [nobjs, nmeta]
        Teff, Feh, etc, etc

    scatter:


    Returns
    -------
    coeff: ndarray
        coefficients of the fit

    MTCinvM: ndarray
        inverse covariance matrix for fit coefficients

    chi: float
        chi-squared at best fit

    logdet_Cinv: float
        inverse of the log determinant of the cov matrice
        :math:`\sum(\log(Cinv))`
    """
    # least square fit
    #pick = logical_and(data[:,1] < np.median(data[:,1]) + np.std(data[:,1])*3. , data[:,1] >  median(data[:,1]) - np.std(data[:,1])*3.)#5*std(data[:,1]) ) 
    Cinv = 1. / (data[:, 2] ** 2 + scatter ** 2)  # invvar slice of data
    M = features
    MTCinvM = np.dot(M.T, Cinv[:, None] * M) # craziness b/c Cinv isnt a matrix
    x = data[:, 1] # intensity slice of data
    MTCinvx = np.dot(M.T, Cinv * x)
    try:
        coeff = np.linalg.solve(MTCinvM, MTCinvx)
    except np.linalg.linalg.LinAlgError:
        print MTCinvM, MTCinvx, data[:,0], data[:,1], data[:,2]
        print features
    assert np.all(np.isfinite(coeff)) 
    chi = np.sqrt(Cinv) * (x - np.dot(M, coeff)) 
    logdet_Cinv = np.sum(np.log(Cinv)) 
    return (coeff, MTCinvM, chi, logdet_Cinv )

def do_one_regression(data, metadata):
    """do_one_regression
    This determines the scatter of the fit at a single wavelength for all stars

    Parameters
    ----------

    data:
    metadata:


    returns
    -------

    """
    ln_s_values = np.arange(np.log(0.0001), 0., 0.5)
    chis_eval = np.zeros_like(ln_s_values)
    for ii, ln_s in enumerate(ln_s_values):
        foo, bar, chi, logdet_Cinv = do_one_regression_at_fixed_scatter(data, metadata, scatter = np.exp(ln_s))
        chis_eval[ii] = np.sum(chi * chi) - logdet_Cinv
    if np.any(np.isnan(chis_eval)):
        s_best = np.exp(ln_s_values[-1])
        return do_one_regression_at_fixed_scatter(data, metadata, scatter = s_best) + (s_best, )
    lowest = np.argmin(chis_eval)
    if lowest == 0 or lowest == len(ln_s_values) + 1:
        s_best = np.exp(ln_s_values[lowest])
        return do_one_regression_at_fixed_scatter(data, metadata, scatter = s_best) + (s_best, )
    ln_s_values_short = ln_s_values[np.array([lowest-1, lowest, lowest+1])]
    chis_eval_short = chis_eval[np.array([lowest-1, lowest, lowest+1])]
    z = np.polyfit(ln_s_values_short, chis_eval_short, 2)
    f = np.poly1d(z)
    fit_pder = np.polyder(z)
    fit_pder2 = pylab.polyder(fit_pder)
    s_best = np.exp(np.roots(fit_pder)[0])
    return do_one_regression_at_fixed_scatter(data, metadata, scatter = s_best) + (s_best, )

def do_regressions(dataall, features):
    """
    This loops through all the regressions = doing the fit at a single wavelength for all stars, for all wavelengths
    """
    nlam, nobj, ndata = dataall.shape
    nobj, npred = features.shape
    featuresall = np.zeros((nlam,nobj,npred))
    featuresall[:, :, :] = features[None, :, :]
    return map(do_one_regression, dataall, featuresall)

def train(dataall, metaall, order, fn, Ametaall, cluster_name, logg_cut=100., teff_cut=0., leave_out=None):
    """
    - `leave out` must be in the correct form to be an input to `np.delete`
    - this is the routine that determines the coefficients from the training data 
    """
   # good = np.logical_and((metaall[:, 1] < logg_cut), (metaall[:,0] > teff_cut) ) 
   # dataall = dataall[:, good]
   # metaall = metaall[good]

    diff_t = np.abs(np.array(metaall[:,0] - Ametaall[:,0]) ) 
    good = np.logical_and((metaall[:, 1] < logg_cut), (diff_t < 600. ) ) 
    dataall = dataall[:, good]
    metaall = metaall[good]

    nstars, nmeta = metaall.shape

    if leave_out is not None: #
        dataall = np.delete(dataall, [leave_out], axis = 1) 
        metaall = np.delete(metaall, [leave_out], axis = 0) 

    offsets = np.mean(metaall, axis=0)
    features = np.ones((nstars, 1))
    if order >= 1:
        features = np.hstack((features, metaall - offsets)) 
    if order >= 2:
        newfeatures = np.array([np.outer(m, m)[np.triu_indices(nmeta)] for m in (metaall - offsets)])
        features = np.hstack((features, newfeatures))

    blob = do_regressions(dataall, features)
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    chisqs = np.array([np.dot(b[2],b[2]) - b[3] for b in blob]) # holy crap be careful
    scatters = np.array([b[4] for b in blob])

    fd = open(fn, "w")
    pickle.dump((dataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisqs), fd)
    fd.close()

## non linear stuff below ##
# returns the non linear function 
#def func(x1, x2, x3, x4, x5, x6, x7, x8, x9, a, b, c):
#    f = (0 
#         + x1*a 
#         + x2*b 
#         + x3*c 
#         + x4* a**2# 
#         + x5 * a * b
#         + x6 * a * c 
#         + x7*b**2
#         + x8  * b * c 
#         + x9*c**2 )
#    return f

# this is the form of the function below of the labels (a,b,c,d) = [Teff, logg , [FeH], age and their coefficients x1-x14
def func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 , x13, x14, a, b, c, d):
    f = (0 
         + x1*a 
         + x2*b 
         + x3*c 
         + x4*d 
         + x5* a**2# 
         + x6 * a * b
         + x7 * a * c
         + x8 * a * d 
         + x9* b**2
         + x10  * b * c 
         + x11  * b * d 
         + x12* c**2  
         + x13 * c * d
         + x14* d**2 )
    return f

## thankyou stack overflow for the example below on how to use the optimse function  
#def nonlinear_invert(f, x1, x2, x3, x4, x5, x6, x7, x8, x9 ,sigmavals):
#    def wrapped_func(observation_points, a, b, c):
#        x1, x2, x3, x4, x5, x6, x7, x8, x9  = observation_points
#        return func(x1, x2, x3, x4, x5, x6, x7, x8, x9,  a, b, c)
# thankyou stack overflow for the example below on how to use the optimse function  
def nonlinear_invert(f, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, sigmavals):
    def wrapped_func(observation_points, a, b, c, d):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14   = observation_points
        return func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,  a, b, c, d)

    xdata = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 ])
    model, cov = opt.curve_fit(wrapped_func, xdata, f, sigma = sigmavals)#absolute_sigma = True)  is not an option in my version of scipy will upgrade scipy
    return model, cov 

def infer_labels_nonlinear(fn_pickle,testdata, fout_pickle, weak_lower,weak_upper):
#def infer_labels(fn_pickle,testdata, fout_pickle, weak_lower=0.935,weak_upper=0.98):
    """
    - this routine determines the labels for a new spectra - can do cuts on the flux for testing if want to 
    best log g = weak_lower = 0.95, weak_upper = 0.98
    best teff = weak_lower = 0.95, weak_upper = 0.99
    best_feh = weak_lower = 0.935, weak_upper = 0.98 
    this returns the parameters for a field of data  - and normalises if it is not already normalised 
    this is slow because it reads a pickle file 
    """
    file_in = open(fn_pickle, 'r') 
    dataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisq = pickle.load(file_in)
    file_in.close()
    nstars = (testdata.shape)[1]
    nlabels = len(labels)
    Params_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, np.shape(coeffs)[1]-1, np.shape(coeffs)[1]-1.))
    covs_all = np.zeros((nstars,nlabels, nlabels))
    for jj in range(0,nstars):
      if np.any(testdata[:,jj,0] != dataall[:, 0, 0]):
          print testdata[range(5),jj,0], dataall[range(5),0,0]
          assert False
      xdata = testdata[:,jj,0]
      ydata = testdata[:,jj,1]
      ysigma = testdata[:,jj,2]
      ydata_norm = ydata  - coeffs[:,0] # subtract the mean 
      f = ydata_norm 
      t,g,feh,age = metaall[:,0], metaall[:,1], metaall[:,2], metaall[:,3]
      x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14 = coeffs[:,0], coeffs[:,1], coeffs[:,2], coeffs[:,3], coeffs[:,4], coeffs[:,5], coeffs[:,6] ,coeffs[:,7], coeffs[:,8], coeffs[:,9], \
      coeffs[:,10], coeffs[:,11], coeffs[:, 12], coeffs[:,13],coeffs[:,14]  
      Cinv = 1. / (ysigma ** 2 + scatters ** 2)
      Params,covs = nonlinear_invert(f, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, 1/Cinv**0.5 ) 
      Params = Params+offsets 
      value_cut = -14
      coeffs_slice = coeffs[:,value_cut:]
      MCM_rotate = np.dot(coeffs_slice.T, Cinv[:,None] * coeffs_slice)
      Params_all[jj,:] = Params 
      MCM_rotate_all[jj,:,:] = MCM_rotate 
      covs_all[jj,:,:] = covs
    file_in = open(fout_pickle, 'w')  
    pickle.dump((Params_all, covs_all),  file_in)
    file_in.close()
    return Params_all , MCM_rotate_all

def infer_labels(fn_pickle,testdata, fout_pickle, weak_lower,weak_upper):
#def infer_labels(fn_pickle,testdata, fout_pickle, weak_lower=0.935,weak_upper=0.98):
    """ 
    - this is the linear case of getting labels for new spectra 
    best log g = weak_lower = 0.95, weak_upper = 0.98
    best teff = weak_lower = 0.95, weak_upper = 0.99
    best_feh = weak_lower = 0.935, weak_upper = 0.98 
    this returns the parameters for a field of data  - and normalises if it is not already normalised 
    this is slow because it reads a pickle file 
    """
    file_in = open(fn_pickle, 'r') 
    dataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisqs = pickle.load(file_in)
    file_in.close()
    nstars = (testdata.shape)[1]
    nlabels = len(labels)
    Params_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, nlabels, nlabels))
    for jj in range(0,nstars):
      if np.any(testdata[:,jj,0] != dataall[:, 0, 0]):
          print testdata[range(5),jj,0], dataall[range(5),0,0]
          assert False
      xdata = testdata[:,jj,0]
      ydata = testdata[:,jj,1]
      ysigma = testdata[:,jj,2]
      ydata_norm = ydata  - coeffs[:,0] # subtract the mean 
      cut_to = shape(metaall)[1]*-1.
      coeffs_slice = coeffs[:,cut_to:]
      #ind1 = np.logical_and(logical_and(dataall[:,jj,0] > 16200., dataall[:,jj,0] < 16500.), np.logical_and(ydata > weak_lower , ydata < weak_upper)) 
      ind1 =  np.logical_and(ydata > weak_lower , ydata < weak_upper)
      Cinv = 1. / (ysigma ** 2 + scatters ** 2)
      MCM_rotate = np.dot(coeffs_slice[ind1].T, Cinv[:,None][ind1] * coeffs_slice[ind1])
      MCy_vals = np.dot(coeffs_slice[ind1].T, Cinv[ind1] * ydata_norm[ind1]) 
      Params = np.linalg.solve(MCM_rotate, MCy_vals)
      Params = Params + offsets 
      print Params
      Params_all[jj,:] = Params 
      MCM_rotate_all[jj,:,:] = MCM_rotate 
    file_in = open(fout_pickle, 'w')  
    pickle.dump((Params_all, MCM_rotate_all),  file_in)
    file_in.close()
    return Params_all , MCM_rotate_all


def lookatfits(fn_pickle, pixelvalues,testdataall): 
  #  """"
  # TEST ROUTINE - PLOTTING ROUTINE
  #  this is to plot the individual pixel fits  on the 6x6 panel 
  #  """"
    file_in = open(fn_pickle, 'r') 
    testdataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisqs = pickle.load(file_in)
    file_in.close()
    axis_t, axis_g, axis_feh = metaall[:,0], metaall[:,1], metaall[:,2]
    nstars = (testdataall.shape)[1]
    offsets = np.mean(metaall, axis=0)
    features = np.ones((nstars, 1))
    features = np.hstack((features, metaall - offsets)) 
    features2 = np.hstack((features, metaall )) 
    for each in pixelvalues:
        flux_val_abs = testdataall[each,:,1]
        flux_val_norm = testdataall[each,:,1] - np.dot(coeffs, features.T)[each,:] 
        coeff = coeffs[each,:] 
        y_feh_abs = coeff[3]*features[:,3] + coeff[0]*features[:,0]
        y_feh_norm = coeff[3]*features[:,3] + coeff[0]*features[:,0]  -(coeff[3]*features2[:,3] + coeff[0]*features2[:,0]) 
        y_g_abs = coeff[2]*features[:,2] + coeff[0]*features[:,0]
        y_g_norm = coeff[2]*features[:,2] + coeff[0]*features[:,0]  - (coeff[2]*features2[:,2] + coeff[0]*features2[:,0]) 
        y_t_abs = coeff[1]*features[:,1] + coeff[0]*features[:,0] 
        y_t_norm = coeff[1]*features[:,1] + coeff[0]*features[:,0] - (coeff[1]*features2[:,1] + coeff[0]*features2[:,0]) 
        for flux_val, y_feh, y_g, y_t, namesave,lab,ylims in zip([flux_val_abs, flux_val_norm], [y_feh_abs,y_feh_norm],[y_g_abs, y_g_norm], [y_t_abs,y_t_norm],['abs','norm'], ['flux','flux - mean'],
                [[-0.2,1.2], [-1,1]] ): 
            y_meandiff = coeff[0] - flux_val 
            fig = plt.figure(figsize = [12.0, 12.0])
            #
            ax = plt.subplot(3,2,1)
            pick = testdataall[each,:,2] > 0.1
            ax.plot(metaall[:,2], flux_val, 'o',alpha =0.5,mfc = 'None', mec = 'r') 
            ax.plot(metaall[:,2][pick], flux_val[pick], 'kx',markersize = 10) 
            ax.plot(metaall[:,2], y_feh, 'k') 
            ind1 = argsort(metaall[:,2]) 
            ax.fill_between(sort(metaall[:,2]), array(y_feh + std(flux_val))[ind1], array(y_feh - std(flux_val))[ind1] , color = 'y', alpha = 0.2)
            ax.set_xlabel("[Fe/H]", fontsize = 14 ) 
            ax.set_ylabel(lab, fontsize = 14 ) 
            ax.set_title(str(np.int((testdataall[each,0,0])))+"  $\AA$")
            ax.set_ylim(ylims[0], ylims[1]) 
            #
            ax = plt.subplot(3,2,2)
            ax.plot(metaall[:,1], flux_val, 'o', alpha =0.5, mfc = 'None', mec = 'b') 
            ax.plot(metaall[:,1][pick], flux_val[pick], 'kx',markersize = 10)  
            ax.plot(metaall[:,1], y_g, 'k') 
            ind1 = argsort(metaall[:,1]) 
            ax.fill_between(sort(metaall[:,1]), array(y_g + std(flux_val))[ind1], array(y_g - std(flux_val))[ind1] , color = 'y', alpha = 0.2)
            ax.set_xlabel("log g", fontsize = 14 ) 
            ax.set_ylabel(lab, fontsize = 14 ) 
            ax.set_title(str(np.int((testdataall[each,0,0])))+"  $\AA$")
            ax.set_ylim(ylims[0], ylims[1]) 
            #
            ax = plt.subplot(3,2,3)
            ax.plot(metaall[:,0], flux_val, 'o',alpha =0.5, mfc = 'None', mec = 'green') 
            ax.plot(metaall[:,0][pick], flux_val[pick], 'kx', markersize = 10) 
            ax.plot(metaall[:,0], y_t, 'k') 
            ind1 = argsort(metaall[:,0]) 
            ax.fill_between(sort(metaall[:,0]), array(y_t + std(flux_val))[ind1], array(y_t - std(flux_val))[ind1] , color = 'y', alpha = 0.2)
            ax.set_xlabel("Teff", fontsize = 14 ) 
            ax.set_ylabel(lab, fontsize = 14 ) 
            ax.set_ylim(ylims[0], ylims[1]) 
            #
            ax = plt.subplot(3,2,4)
            diff_flux = coeffs[each,0] - testdataall[each,:,1] 
            xrange1 = arange(0,shape(testdataall)[1],1) 
            ind1 = argsort(metaall[:,2]) 
            ind1_pick = argsort(metaall[:,2][pick]) 
            ax.plot(xrange1, (coeffs[each,0] - testdataall[each,:,1])[ind1], 'o',alpha = 0.5, mfc = 'None', mec = 'grey') 
            ax.plot(xrange1[pick], (coeffs[each,0] - testdataall[each,:,1][pick])[ind1_pick], 'kx',markersize = 10) 
            ax.fill_between(xrange1, array(mean(diff_flux) + std(diff_flux)), array(mean(diff_flux) - std(diff_flux))  , color = 'y', alpha = 0.2)
            ax.set_xlabel("Star Number (increasing [Fe/H])", fontsize = 14 ) 
            ax.set_ylabel("flux star - mean flux", fontsize = 14 ) 
            ax.set_ylim(-1.0, 1.0) 
            #
            ax = plt.subplot(3,2,5)
            for indx, color, label in [
                                       ( 1, "g", "Teff"),
                                       ( 2, "b", "logg"),
                                       ( 3, "r", "FeH")]:
              _plot_something(ax, testdataall[:, 0, 0][each-10:each+10], coeffs[:, indx][each-10:each+10], covs[:, indx, indx][each-10:each+10], color, label=label)
            ax.axvline(testdataall[:,0,0][each],color = 'grey') 
            ax.axhline(0,color = 'grey',linestyle = 'dashed') 
            ax.set_xlim(testdataall[:,0,0][each-9], testdataall[:,0,0][each+9]) 
            ax.legend(loc = 4,fontsize  = 10) 
            ax.set_xlabel("Wavelength $\AA$", fontsize = 14 ) 
            ax.set_ylabel("coeffs T,g,FeH", fontsize = 14 ) 
            #
            ax = plt.subplot(3,2,6)
            _plot_something(ax, testdataall[:, 0, 0][each-10:each+10], coeffs[:, 0][each-10:each+10], covs[:, 0, 0][each-10:each+10], 'k', label='mean')
            ax.set_ylim(0.6,1.1) 
            ax.set_xlim(testdataall[:,0,0][each-9], testdataall[:,0,0][each+9]) 
            ax.legend(loc = 4,fontsize  = 10) 
            ax.axvline(testdataall[:,0,0][each],color = 'grey') 
            ax.axhline(0,color = 'grey',linestyle = 'dashed') 
            ax.set_xlabel("Wavelength $\AA$", fontsize = 14 ) 
            ax.set_ylabel("Mean flux", fontsize = 14 ) 

            savefig3(fig, str(each)+"_"+str(namesave) , transparent=False, bbox_inches='tight', pad_inches=0.5)
            fig.clf()
       # return 


def _plot_something(ax, wl, val, var, color, lw=2, label=""):
    """
    This routine is for plotting 
    """
    factor = 1.
    if label == "Teff": factor = 1000. # yes, I feel dirty; MAGIC
    sig = np.sqrt(var)
    ax.plot(wl, factor*(val+sig), color=color, lw=lw, label=label)
    ax.plot(wl, factor*(val-sig), color=color, lw=lw) 
    ax.fill_between(wl, factor*(val+sig), factor*(val-sig), color = color, alpha = 0.2) 
    return None
  
def savefig3(fig, prefix, **kwargs):
 #   for suffix in (".png"):
    suffix = ".png"
    print "writing %s" % (prefix + suffix)
    fig.savefig(prefix + suffix)#, **kwargs)
    close() 

def leave_one_cluster_out_xval(cluster_information):
    # this is done in the fitspectra.py code- can look at this if want to implement it later and copy it directly across
    dataall, metaall, labels = get_normalized_training_data()
    for jj, cluster_indx in enumerate(clusters):
        cluster_indx = something
        pfn = "coeffs_%03d.pickle" % (jj)
        # read_and_train(dataall, .., pfn, leave_out=cluster_indx)
        # infer_labels(pfn, dataall[:, cluster_indx], ofn)
        # plotting...

if __name__ == "__main__":
    dataall, metaall, labels, Ametaall, cluster_name  = get_normalized_training_data()
    fpickle = "coeffs.pickle" 
    if not glob.glob(fpickle):
        train(dataall, metaall, 1,  fpickle, Ametaall, cluster_name, logg_cut= 40.,teff_cut = 0.)
    fpickle2 = "coeffs_2nd_order.pickle"
    if not glob.glob(fpickle2):
        train(dataall, metaall, 2,  fpickle2, Ametaall, cluster_name, logg_cut= 40.,teff_cut = 0.)
    self_flag = 2
    if self_flag < 1:
        a = open('all.txt', 'r') 
        a = open('all_test2.txt', 'r') 
        al = a.readlines()
        bl = []
        for each in al:
            bl.append(each.strip()) 
        for each in bl: 
            testfile = each
            field = testfile.split('.txt')[0]+'_' #"4332_"
            testdataall = get_normalized_test_data(testfile) # if flag is one, do on self 
            testmetaall, inv_covars = infer_labels_nonlinear("coeffs_2nd_order.pickle", testdataall, field+"tags.pickle",-10.94,10.99) 
    
    if self_flag == 1:
      field = "self_"
      file_in = open('normed_data.pickle', 'r') 
      testdataall, metaall, labels = pickle.load(file_in)
      lookatfits('coeffs.pickle',[1002],testdataall)
      file_in.close() 
      testmetaall, inv_covars = infer_labels("coeffs.pickle", testdataall, field+"tags.pickle",-10.980,11.43) 
    if self_flag == 2:
      field = "self_2nd_order_"
      file_in = open('normed_data.pickle', 'r') 
      testdataall, metaall, labels, Ametaall, cluster_name = pickle.load(file_in)
      file_in.close() 
      testmetaall, inv_covars = infer_labels_nonlinear("coeffs_2nd_order.pickle", testdataall, field+"tags.pickle",-10.950,10.99)

    # Calculate the chi squared of the fit
    nstars = (testdataall.shape)[1]
    nlabels = len(labels)
    print "Number of stars: %s" %str(nstars)
    print "Number of labels: %s" %str(nlabels)
    Params_all = np.zeros((nstars, nlabels))
    red_chisq = get_goodness_fit('coeffs_2nd_order.pickle', 'normed_data', Params_all)
 
