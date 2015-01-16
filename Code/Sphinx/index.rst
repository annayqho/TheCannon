*****************************************
*The Cannon*: Data-Driven Stellar Labels
*****************************************

Introduction
============

This is the software package used for *The Cannon*,
a data-driven approach to determining stellar labels (parameters
and abundances) for a vast set of stellar spectra. This version is tailored 
specifically for APOGEE spectra.

A brief overview of *The Cannon* and the associated software package is below. 
For more details on the method and its successful application to APOGEE DR10
spectra, see Ness et al. 2015.

Introduction to *The Cannon* 
----------------------------

*The Cannon* has two fundamental steps that together constitutes a 
process of *label transfer.* 

1. The *Training Step*: *reference objects* are a subset of spectra in the 
   survey for which corresponding stellar labels are known with high fidelity, 
   for calib reasons or otherwise. Using both the spectra and labels for 
   these objects, *The Cannon* solves for a flexible model that describes 
   how the flux in every pixel of any given continuum-normalized spectrum 
   depends on labels. 
   
2. The *Test Step*: the model found in Step 1 is assumed to hold for all of 
   the objects in the survey, including those outside the training set 
   (dubbed *survey objects*). Thus, the spectra of the survey objects and 
   the model allow us to solve for - or infer - the labels of the survey 
   objects. 

A Word on Spectra
-----------------

*The Cannon* expects all spectra - for reference and survey objects - 
to be continuum-normalized in a consistent way, and sampled on a consistent
rest-frame wavelength grid, with the same line-spread function. It also
assumes that the flux variance, from photon noise and other sources, is 
known at each spectral pixel of each spectrum.

Overview of *The Cannon* Software
---------------------------------

This software package breaks up *The Cannon* into the following steps and 
features.

#. Construct a training set from APOGEE files
   
   * ``read_apogee``: retrieve continuum-normalized training spectra 
   and (optional) select a subset of spectra.
   * ``read_labels``: retrieve training label names and values
   and (optional) select a subset of labels.
   * Run a set of diagnostics on the training set

#. Construct a test set from APOGEE files

   * ``read_apogee``: retrieve continuum-normalized test spectra 
     and (optional) select a subset of spectra.

#. Step 1 of The Cannon: fit for a model

   * Run a set of diagnostics on the model

#. Step 2 of The Cannon: infer labels for all test objects

   * Run a set of diagnostics on the inferred labels
   * Run a set of diagnostics on the best-fit spectra

Using *The Cannon*
==================

The details of using The Cannon package are provided in the following 
sections, along with the following example: 553 open and globular cluster stars 
from APOGEE DR10 as the training set, and, for simplicity, the same set of stars
as the test set. 

Step 1: Construct a training set from APOGEE files (``apogee.py``) 
------------------------------------------------------------------

The training set is a set of stars from the survey under consideration
for which the user has spectra and also high-fidelity labels (that is,
stellar parameters and element abundances that are deemed both accurate
and precise.) The set of reference objects is critical, as the label 
transfer to the survey objects can only be as good as the quality of the
training set. 

The user must construct the following inputs: 

1. a list of filenames corresponding to the training data 
   (in this case, APOGEE .fits files) 
2. a .txt file containing training labels in an ASCII table. 

The following requirements govern (2):
1. The first row must be strings corresponding to the names of the labels 
   and must not contain any '/'s 
2. The first column must be string corresponding to the stellar IDs
3. The remaining entries must be floats corresponding to the label values

In our example, the label file is called ``traininglabels.txt`` and the ID 
column happens to correspond to the file names that we want to read spectra 
from.

    >>> import numpy as np
    >>> readin = "traininglabels.txt"
    >>> IDs = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
    >>> filenames1 = []
    >>> for i in range(0, len(IDs)): #incorporate file location info
        ...filename = '/home/annaho/AnnaCannon/Data/APOGEE_Data' + IDs[i][1:]
        ...filenames1.append(filename)

Once the file list is created, the ``get_spectra`` method can be               
used to put the spectrum information into the correct format.

    >>> from read_apogee import get_spectra
    >>> spectra, SNRs = get_spectra(filenames1) 

Now the ``get_training_labels`` method is used to retrieve IDs, label names, 
and label values.

    >>> from read_labels import get_training_labels
    >>> IDs, all_label_names, all_label_values = get_training_labels(readin)

(Optional) The user can choose to select some subset of the training labels 
by creating a mask corresponding to the desired column indices. 
In this example, we select Teff, logg, and [Fe/H].  
    
    >>> cols = [1, 3, 5]
    >>> colmask = np.zeros(len(all_label_names), dtype=bool)
    >>> colmask[cols] = 1


    >>> label_names = [all_label_names[i] for i in cols]
    >>> label_values = all_label_values[:,colmask]

The user can also (if desired) select some subset of the training objects, by 
imposing physical cutoffs. Here, we select data with physical 
Teff and logg cutoffs.

    >>> Teff = label_values[:,0]
    >>> Teff_corr = all_label_values[:,2]
    >>> diff_t = np.abs(Teff-Teff_corr)
    >>> diff_t_cut = 600.
    >>> logg = label_values[:,1]
    >>> logg_cut = 100.
    >>> bad = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
    >>> IDs = IDs[bad]
    >>>

    >>> dataset import Dataset
    >>> fts_trainingset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = [])
    >>> vesta_trainingset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = [])
    >>> cluster_trainingset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = [])
    >>> trainingset = mergesets(fts_trainingset, vesta_trainingset, cluster_trainingset)

There are a few ways to examine the dataset. You can retrieve the spectra
as follows:

>>> pixels = trainingset.spectra[:,:,0]
>>> fluxes = trainingset.spectra[:,:,1]
>>> fluxerrs = trainingset.spectra[:,:,2]
    
Step 3: Construct Test Set
---------------------------

    >>> testset = Dataset(objectIDs = [], spectra = [], labelnames = [], labelvals = None)

Step 4: *The Cannon* Step 1 - Generate Model
---------------------------------------------

    >>> from spectral_model import SpectralModel
    >>> model = SpectralModel(label_names, modeltype) 
    >>> model.train(trainingset)

Step 5: *The Cannon* Step 2 - Infer Labels
-------------------------------------------

    >>> from cannon_labels import CannonLabels
    >>> labels = CannonLabels(label_names)
    >>> labels.solve(model, testset)
