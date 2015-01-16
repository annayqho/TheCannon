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

Overview of *The Cannon* Software
---------------------------------

This software package breaks up *The Cannon* into the following steps and 
features.

#. Construct a training set
   
   * Retrieve continuum-normalized training spectra
   * Retrieve training labels
   * (Optional) Select a subset of labels and spectra
   * Run a set of diagnostics on the training set

#. Construct a test set

   * Retrieve continuum-normalized test spectra
   * (Optional) Select a subset of spectra

#. Step 1 of The Cannon: fit for a model

   * Run a set of diagnostics on the model

#. Step 2 of The Cannon: infer labels for all test objects

   * Run a set of diagnostics on the inferred labels
   * Run a set of diagnostics on the best-fit spectra

Using *The Cannon*
==================

The details of using The Cannon package are provided in the following 
sections, along with an example.

Step 1: Prepare Data (``prepdata.py``) 
---------------------------------------

*The Cannon* expects all spectra - for reference and survey objects - 
to be continuum-normalized in a consistent way, and sampled on a consistent
rest-frame wavelength grid, with the same line-spread function. It also
assumes that the flux variance, from photon noise and other sources, is 
known at each spectral pixel of each spectrum.

Preparing data thus involves: putting spectra into a 3D array
(nstars, npixels, 3), putting label names into an array (nlabels),
and putting training label values into a 2D array (nstars, nlabels).
The user is left to do this him or herself, but we include some basic 
methods and functionalities in ``prep_data.py`` that might be helpful.

Step 2: Construct Training Set
-------------------------------

The training set is a set of stars from the survey under consideration
for which the user has spectra and also high-fidelity labels (that is,
stellar parameters and element abundances that are deemed both accurate
and precise.) The set of reference objects is critical, as the label 
transfer to the survey objects can only be as good as the quality of the
training set. 

Labels for a training set will not necessarily come from the same data,
and therefore spectra will not necessarily be in the same data format.
For flexibility, *The Cannon* software package allows the user to construct
training subsets, one data type each, and then merge the subsets.

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
