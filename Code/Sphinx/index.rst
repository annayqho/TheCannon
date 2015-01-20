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

*The Cannon* has two fundamental steps that together constitute a 
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
   
   * ``get_spectra``: read spectra, continuum-normalize
   * ``get_training_labels``: retrieve stellar IDs, training label names and values
   * ``choose_labels``: (optional) select a subset of labels
   * ``choose_spectra``: (optional) select a subset of spectra  
   * ``training_set_diagnostics``: (optional) run a set of diagnostics 
     on the training set

#. Construct a test set from APOGEE files

   * ``get_spectra``: read spectra, continuum-normalize
   * ``choose_spectra``: (optional) select a subset of spectra

#. The Cannon Step 1: Fit Model

   * ``train_model``: solve for model
   * ``model_diagnostics``: run a set of diagnostics on the model

#. Step 2 of The Cannon: infer labels for all test objects

   * ``infer_labels``: infer labels using model
   * ``test_set_diagnostics``: run a set of diagnostics on the inferred labels

#. Cannon-generated spectra (``cannon_spectra``)

   * ``draw_spectra`` in ``cannon_spectra``
   * ``diagnostics`` in ``cannon_spectra``

Using *The Cannon*
==================

The details of using this package are provided in the following 
sections, along with an example of usage. In the example, the training set
consists of 553 open and globular cluster stars from APOGEE DR10 and, 
for simplicity, the same set of stars as the test set. 

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
2. The first column must be strings corresponding to the stellar IDs
3. The remaining entries must be floats corresponding to the label values

Reading spectra (``get_spectra``)
+++++++++++++++++++++++++++++++++

We construct the first input: the list of filenames corresponding to the 
training data (in this case, APOGEE .fits files). In our example, the filenames
happen to be the first column in the training labels text file, 
``traininglabels.txt``. So we simply read the first column of this file.

    >>> import numpy as np
    >>> readin = "traininglabels.txt"
    >>> IDs = np.loadtxt(readin, usecols=(0,), dtype='string', unpack=1)
    >>> filenames1 = []
    >>> for i in range(0, len(IDs)): #incorporate file location info
    >>> ....filename = '/home/annaho/AnnaCannon/Data/APOGEE_Data' + IDs[i][1:]
    >>> ....filenames1.append(filename)

Once the file list is created, the ``get_spectra`` method can be               
used to continuum-normalize the spectrum information and put it 
into the correct format. ``get_spectra`` also returns the fitted
continua (Chebyshev polynomial) and a list of SNR values for the 
spectra.

    >>> from read_apogee import get_spectra
    >>> lambdas, normalized_spectra, continua, SNRs = get_spectra(filenames1)

Reading labels (``get_training_labels``)
++++++++++++++++++++++++++++++++++++++++

We construct the second input: the .txt file containing training labels in an 
ASCII table, with requirements described above. In this example, the .txt file
is called ``traininglabels.txt``. The method ``get_training_labels`` is used 
to retrieve object IDs, label names, and label values.

    >>> from read_labels import get_training_labels
    >>> IDs, all_label_names, all_label_values = get_training_labels(readin)

Creating & tailoring a ``Dataset`` object (``choose_labels``, ``choose_spectra``)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A ``Dataset`` object (``dataset.py``) is initialized. 

    >>> from dataset import Dataset
    >>> training_set = Dataset(IDs=IDs, SNRs=SNRs, lambdas=lambdas,
    >>> ....spectra=normalized_spectra, label_names=all_label_names, 
    >>> ....label_vals=all_label_values)

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

    >>> Teff = training_set.label_vals[:,0]
    >>> Teff_corr = all_label_values[:,2]
    >>> diff_t = np.abs(Teff-Teff_corr)
    >>> diff_t_cut = 600.
    >>> logg = training_set.label_vals[:,1]
    >>> logg_cut = 100.
    >>> mask = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
    >>> training_set.choose_spectra(mask)

Training set diagnostics (training_set_diagnostics)
+++++++++++++++++++++++++++++++++++++++++++++++++++

Now, the training set has been constructed. To let the user examine whether 
things are going smoothly, *The Cannon* can print out a set of training set 
diagnostics.

    >>> from dataset import training_set_diagnostics
    >>> training_set_diagnostics(training_set)

The output of these diagnostics, with examples, are listed below.

1. A histogram showing the distribution of SNR in the training set

.. image:: trainingset_SNRdist.png
    :width: 400pt

2. A histogram for each label showing its coverage in label space

.. image:: trainingset_labeldist_Teff.png
    :width: 400pt
   
3. A "triangle plot" that shows every label plotted against every other 

.. image:: trainingset_labels_triangle.png
    :width: 400pt
   
Step 2: Construct a test set from APOGEE files
----------------------------------------------

To construct the test set, the user would ordinarily go through a process 
identical to that for the training set, except without reading in the 
training labels file. 
In this case, for simplicity, we use the training set as our test set. 

    >>> test_set = Dataset(IDs=training_set.IDs, SNRs=training_set.SNRs,
    >>> ....lambdas=lambdas, spectra=training_set.spectra,
    >>> ....label_names=training_set.label_names)

Step 3: *The Cannon* Step 1 - Fit Model (``train_model``, ``model_diagnostics``)
--------------------------------------------------------------------------------

Now, we use our training set to fit for the model.

    >>> from cannon1_train_model import train_model
    >>> model = train_model(training_set)

To let the user examine whether things are going smoothly, *The Cannon* can 
print out a set of model diagnostics.

    >>> from cannon1_train_model import model_diagnostics
    >>> model_diagnostics(training_set, model)

The output of these diagnostics with sample plots are listed below.

1. Plot of the baseline spectrum (0th order coefficients) as a 
   function of wavelength.

.. image:: baseline_spec_with_cont_pix.png
    :width: 400pt

2. Plot of the leading coefficients of each label as a function 
   of wavelength

.. image:: leading_coeffs.png
    :width: 400pt

3. Histogram of the reduced chi squareds of the fits (normalized by DOF, 
   where DOF = npixels-nlabels)

.. image:: modelfit_redchisqs.png
    :width: 400pt

Step 4: *The Cannon* Step 2 - Infer Labels (``infer_labels``, ``test_set_diagnostics``)
---------------------------------------------------------------------------------------

Now, we use the model to infer labels for the survey objects and 
update the test_set object.

    >>> from cannon2_infer_labels import infer_labels
    >>> test_set, covs = infer_labels(model, test_set)

To let the user examine whether things are going smoothly, *The Cannon* can 
print out a set of test set diagnostics.

    >>> from dataset import test_set_diagnostics
    >>> test_set_diagnostics(training_set, test_set)

The output of these diagnostics with sample plots are listed below.

1. For each label, a list of flagged stars for which test labels are 
   over 2-sigma away from training labels
2. Triangle plot, each test label plotted against every other test label

.. image:: testset_labels_triangle.png
    :width: 400pt

3. 1-1 plots, for each label, training values plotted against test values

.. image:: 1to1_labelTeff.png
    :width: 300pt

.. image:: 1to1_labellogg.png
    :width: 300pt

.. image:: 1to1_label[MH].png
    :width: 300pt

Cannon Spectra (``draw_spectra``, ``diagnostics``)
--------------------------------------------------

Now that we have the model and labels for the test objects, ``The Cannon`` can
"draw" spectra for each test object.

    >>> from cannon_spectra import draw_spectra
    >>> cannon_set = draw_spectra(model, test_set)

We can now perform a final set of diagnostic checks.

    >>> from cannon_spectra import diagnostics
    >>> diagnostics(cannon_set, test_set, model)

The output of these diagnostics with sample plots are listed below.

1. A directory called SpectrumFits containing (for 10 randomly-selected stars)
   the Cannon fitted spectra overlaid with the 'true' (data) spectra, 
   as well as the two compared in a 1-to-1 plot.

.. image:: Star500.png

2. For each label, the residuals of the spectra fits stacked and sorted 
   by that label. If the functional form of the model is comprehensive enough,
   then this should look like noise and there should be no systematic structure.

.. image:: residuals_sorted_by_Teff.png
