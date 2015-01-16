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

#. Construct a training set from APOGEE files
   
   * ``read_apogee``: read spectra, continuum-normalize
   * ``read_labels``: retrieve stellar IDs, training label names and values
   * ``dataset``: (optional) select a subset of labels
   * ``dataset``: (optional) select a subset of spectra  
   * ``dataset``: (optional) run a set of diagnostics on the training set

#. Construct a test set from APOGEE files

   * ``read_apogee``: read spectra, continuum-normalize
   * ``dataset``: (optional) select a subset of spectra

#. The Cannon Step 1: Fit Model

   * ``train_model`` in ``cannon1_train_model``: solve for model
   * ``model_diagnostics`` in ``cannon1_train_model``: run a set of 
     diagnostics on the model

#. Step 2 of The Cannon: infer labels for all test objects

   * Run a set of diagnostics on the inferred labels
   * Run a set of diagnostics on the best-fit spectra

Using *The Cannon*
==================

The details of using The Cannon package are provided in the following 
sections, along with the following example: 553 open and globular cluster stars 
from APOGEE DR10 as the training set, and, for simplicity, the same set of stars
as the test set. 

Step 1: Construct a training set from APOGEE files 
--------------------------------------------------

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

Reading spectra (``read_apogee.py``)
++++++++++++++++++++++++++++++++++++

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
used to continuum-normalize the spectrum information and put it 
into the correct format. 

    >>> from read_apogee import get_spectra
    >>> lambdas, normalized_spectra, continua, SNRs = get_spectra(filenames1) 

Reading labels (``read_labels.py``)
+++++++++++++++++++++++++++++++++++

Now the ``get_training_labels`` method is used to retrieve IDs, label names, 
and label values.

    >>> from read_labels import get_training_labels
    >>> IDs, all_label_names, all_label_values = get_training_labels(readin)

Creating & tailoring a Dataset object (``dataset.py``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++

A ``Dataset`` object (``dataset.py``) is initialized. 

    >>> from dataset import Dataset
    >>> training_set = Dataset(IDs=IDs, SNRs=SNRs, spectra=normalized_spectra, 
    label_names=all_label_names, label_values=all_label_values)

(Optional) The user can choose to select some subset of the training labels 
by creating a list of the desired column indices. 
In this example, we select Teff, logg, and [Fe/H] which correspond to 
columns 1, 3, and 5.   
    
    >>> cols = [1, 3, 5]
    >>> training_set.choose_labels(cols)

(Optional) The user can also select some subset of the training objects 
(for example, by imposing physical cutoffs) by constructing a mask where 
1 = keep this object, and 0 = remove it. Here, we select data using physical 
Teff and logg cutoffs.

    >>> Teff = training_set.label_values[:,0]
    >>> Teff_corr = all_label_values[:,2]
    >>> diff_t = np.abs(Teff-Teff_corr)
    >>> diff_t_cut = 600.
    >>> logg = training_set.label_values[:,1]
    >>> logg_cut = 100.
    >>> mask = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
    >>> training_set.choose_spectra(mask)

Training set diagnostics
++++++++++++++++++++++++

Now, the training set has been constructed. To let the user examine whether 
things are going smoothly, *The Cannon* can print out a set of training set 
diagnostics.

    >>> from dataset import training_set_diagnostics
    >>> training_set_diagnostics(training_set)

The output of these diagnostics are:

1. A histogram showing the distribution of SNR in the training set
2. A histogram for each label showing its coverage in label space
3. A "triangle plot" that shows every label plotted against every other 

Sample output plots below.

.. image:: trainingset_SNRdist.png
    :width: 400pt

.. image:: trainingset_labeldist_Teff.png
    :width: 400pt

.. image:: trainingset_labels_triangle.png
    :width: 400pt

Step 2: Construct a test set from APOGEE files
----------------------------------------------

Ordinarily, the user would go through a process identical to that for the 
training set, except without reading in the training labels file. In this case, 
for simplicity, we use the training set as our test set, so that our results 
simply show that *The Cannon* can return good labels for the set it trained on.

    >>> test_set = Dataset(IDs=training_set.IDs, SNRs=training_set.SNRs, 
    spectra=training_set.spectra, label_names=training_set.label_names)

Step 3: *The Cannon* Step 1 - Fit Model (``cannon1_train_model``)
-----------------------------------------------------------------

Now, we use our training set to fit for the model.

    >>> from cannon1_train_model import train_model
    >>> model, label_vector = train_model(training_set)

To let the user examine whether things are going smoothly, *The Cannon* can 
print out a set of model diagnostics.

    >>> from cannon1_train_model import model_diagnostics
    >>> model_diagnostics(lambdas, training_set.label_names, model)

The output of these diagnostics are:

1. Plot of the baseline spectrum (0th order coefficients) as a 
   function of wavelength.
2. Plot of the leading coefficients of each label as a function 
   of wavelength
3. Histogram of the reduced chi squareds of the fits (normalized by DOF, 
   where DOF = npixels-nlabels)

Sample output plots below.

.. image:: baseline_spec_with_cont_pix.png
    :width: 400pt

.. image:: leading_coeffs.png
    :width: 400pt

.. image:: modelfit_redchisqs.png
    :width: 400pt

Step 4: *The Cannon* Step 2 - Infer Labels (``cannon2_infer_labels``)
---------------------------------------------------------------------

Now, we use the model to infer labels for the survey objects.

    >>> from cannon2_infer_labels import infer_labels
    >>> cannon_labels, MCM_rotate, covs = infer_labels(model, test_set)

We update the test objects accordingly.
    
    >>> test_set.set_label_values(cannon_labels)

To let the user examine whether things are going smoothly, *The Cannon* can 
print out a set of test set diagnostics.

    >>> from dataset import test_set_diagnostics
    >>> test_set_diagnostics(training_set, test_set)

The output of these diagnostics are:

1. For each label, a list of flagged stars for which test labels are 
   over 2-sigma away from training labels
2. Triangle plot, each test label plotted against every other test label
3. 1-1 plots, for each label, training values plotted against test values

Sample output plots below.

.. image:: testset_labels_triangle.png
    :width: 400pt

.. image:: 1to1_labelTeff.png
    :width: 400pt

.. image:: 1to1_labellogg.png
    :width: 400pt

.. image:: 1to1_label[MH].png
    :width: 400pt

Cannon Spectra (``draw_spectra``)
---------------------------------

Now that we have the model and labels for the test objects, we can in 
principle "draw" spectra for each test object.

    >>> from cannon_spectra import draw_spectra
    >>> cannon_set = draw_spectra(label_vector, test_set)

We can now perform a final set of diagnostic checks.

    >>> from cannon_spectra import diagnostics
    >>> diagnostics(cannon_set, model, test_set)

