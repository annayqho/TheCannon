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

1. The *Training Step*: *reference stars* are a subset of the 
   survey for which labels are known with high fidelity, 
   for calib reasons or otherwise. Using both the spectra and labels for 
   these objects, *The Cannon* solves for a flexible model that describes 
   how the flux in every pixel of any given continuum-normalized spectrum 
   depends on labels. 
   
2. The *Test Step*: the model found in Step 1 is assumed to hold for all of 
   the objects in the survey, including those outside the reference stars 
   (dubbed *survey stars*). Thus, the spectra of the survey stars and 
   the model allow us to solve for - or infer - the labels of the survey 
   stars. 


Overview of *The Cannon* Software
---------------------------------

This software package breaks up *The Cannon* into the following steps and methods.

#. Construct reference stars from APOGEE files
   
   * ``get_spectra``: read spectra, continuum-normalize
   * ``get_reference_labels``: retrieve stellar IDs, reference label names and values
   * ``choose_labels``: (optional) select a subset of labels
   * ``choose_objects``: (optional) select a subset of objects  
   * ``dataset_prediagnostics``: (optional) run a set of diagnostics 
     on the reference stars

#. Construct test stars from APOGEE files

   * ``get_spectra``: read spectra, continuum-normalize
   * ``choose_objects``: (optional) select a subset of spectra

#. *The Cannon*'s Training Step: Fit Model

   * ``train_model``: solve for model
   * ``model_diagnostics``: run a set of diagnostics on the model

#. *The Cannon*'s Test Step: Infer Labels

   * ``infer_labels``: infer labels using model
   * ``dataset_postdiagnostics``: run a set of diagnostics on the inferred labels

#. Cannon-generated spectra (``spectral_model``)

   * ``draw_spectra`` in ``spectral_model``
   * ``diagnostics`` in ``spectral_model``

Using *The Cannon*
==================

The details of using this package are provided in the following 
sections, along with an example of usage. In the example, the reference stars
consists of 553 open and globular cluster stars from APOGEE DR10 and, 
for simplicity, the same set of stars as the survey stars. 

Step 1: Construct a set of reference object from APOGEE files 
-------------------------------------------------------------

The reference stars in the survey under consideration
are those which the user has spectra and also high-fidelity labels (that is,
stellar parameters and element abundances that are deemed both accurate
and precise.) The set of reference stars is critical, as the label 
transfer to the survey stars can only be as good as the quality of the
reference stars. 

The user must construct a .txt file containing reference labels in an ASCII table,
according to the following requirements:

1. The first row must be strings corresponding to the names of the labels 
   in a LaTeX compilable format (ex. T_{eff} for effective temperature)
2. The first column must be strings corresponding to the filenames with the raw data
3. The remaining entries must be floats corresponding to the label values

1a. Reading spectra (``get_spectra``)
+++++++++++++++++++++++++++++++++++++

The ``get_spectra`` method extracts the spectrum information from the 
raw data files (SNR, fluxes, uncertainties) and applies a continuum 
normalization by fitting a Chebyshev polynomial. The user specifies the
directory in which all of the data files are stored. In this example, the
directory is called ``Data``. 

    >>> from read_apogee import get_spectra
    >>> lambdas, norm_fluxes, norm_ivars, SNRs = get_spectra("Data")

1b. Reading labels (``get_reference_labels``)
+++++++++++++++++++++++++++++++++++++++++++++

This step assumes that a .txt file containing reference labels has been prepared
in the correct format (as described above). In this example, the .txt file
is called ``reference_labels_update.txt``. The method ``get_reference_labels`` 
is used to retrieve object IDs, label names, and label values.

    >>> from read_labels import get_reference_labels
    >>> IDs, all_label_names, all_label_values = get_reference_labels(
    >>> ..."reference_labels_update.txt")

1c. Creating & tailoring a ``Dataset`` object (``choose_labels``, ``choose_spectra``)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A ``Dataset`` object (``dataset.py``) is initialized. 

    >>> from dataset import Dataset
    >>> reference_set = Dataset(IDs=IDs, SNRs=SNRs, lams=lambdas, 
    >>> ...fluxes = norm_fluxes, ivars = norm_ivars, label_names=all_label_names,
    >>> ...label_vals=all_label_values)

(Optional) The user can choose to select some subset of the reference labels 
by creating a list of the desired column indices. 
In this example, we select Teff, logg, and [Fe/H] which correspond to 
columns 1, 3, and 5.   
    
    >>> cols = [1, 3, 5]
    >>> reference_set.choose_labels(cols)

(Optional) The user can also select some subset of the reference objects 
(for example, by imposing physical cutoffs) by constructing a mask where 
1 = keep this object, and 0 = remove it. Here, we select data using physical 
Teff and logg cutoffs.

    >>> import numpy as np
    >>> Teff = reference_set.label_vals[:,0]
    >>> Teff_corr = all_label_values[:,2]
    >>> diff_t = np.abs(Teff-Teff_corr)
    >>> diff_t_cut = 600.
    >>> logg = reference_set.label_vals[:,1]
    >>> logg_cut = 100.
    >>> mask = np.logical_and((diff_t < diff_t_cut), logg < logg_cut)
    >>> reference_set.choose_objects(mask)

Step 2: Construct a set of test objects from APOGEE files
----------------------------------------------------------

To construct the test set, the user would ordinarily go through a process 
identical to that for the reference set, except without reading in the 
reference labels file. 
In this case, for simplicity, we use the reference set as our test set. 

    >>> test_set = Dataset(IDs=reference_set.IDs, SNRs=reference_set.SNRs,
    >>> ...lams=lambdas, fluxes=reference_set.fluxes, ivars = reference_set.ivars,
    >>> ...label_names=reference_set.label_names)

Dataset Prediagnostics (dataset_prediagnostics)
-----------------------------------------------

Now that the reference and test sets have been constructed, we can examine 
whether things are going smoothly through a set of diagnostic plots. 

    >>> from dataset import dataset_prediagnostics
    >>> dataset_diagnostics(reference_set, test_set)

The output of these diagnostics, with examples, are listed below.

1.1) A histogram showing the distribution of SNR in the reference set overplotted
with the distribution of SNR in the test set.

.. image:: SNRdist.png

1.2) A "triangle plot" that shows the distribution of every label as well as 
every label plotted against every other 

.. image:: survey_labels_triangle.png

Step 3: *The Cannon*'s Training Step (``train_model``, ``model_diagnostics``)
-----------------------------------------------------------------------------

Now, we use our reference set to fit for the model.

    >>> from cannon1_train_model import train_model
    >>> model = train_model(reference_set)

To let the user examine whether things are going smoothly, *The Cannon* can 
print out a set of model diagnostics.

    >>> from cannon1_train_model import model_diagnostics
    >>> model_diagnostics(reference_set, model)

The output of these diagnostics with sample plots are listed below.

3.1) Plot of the baseline spectrum (0th order coefficients) as a 
function of wavelength, with continuum pixels overlaid. Ten plots are
produced, each showing 10% of the spectrum. Examples are shown below
as an animated .gif:

.. image:: baseline_spec_with_cont_pix.gif

3.2) Plot the leading coefficients of each label and scatter
as a function of wavelength

.. image:: leading_coeffs.png

.. image:: leading_coeffs_triangle.png

3.3) Histogram of the chi squareds of the fits, with a dotted line corresponding
to the number of degrees of freedom. 

.. image:: modelfit_chisqs.png


Step 4: *The Cannon*'s Test Step (``infer_labels``, ``test_set_diagnostics``)
-----------------------------------------------------------------------------

Now, we use the model to infer labels for the survey objects and 
update the test_set object.

    >>> from cannon2_infer_labels import infer_labels
    >>> test_set, covs = infer_labels(model, test_set)

Now that the labels have been inferred, *The Cannon* can run another set of 
diagnostics.

    >>> from dataset import dataset_postdiagnostics
    >>> dataset_diagnostics(reference_set, test_set)

The output of these diagnostics with sample plots are listed below.

4.1) One text file for each label, with a list of flagged stars. Flagged stars
are those whose output labels lie over 2-sigma away from the original reference
label. In other words, this is a warning that *The Cannon* has extrapolated
outside of the reference label space. 

4.2) Triangle plot, each test label plotted against every other test label

.. image:: survey_labels_triangle.png

4.3) 1-1 plots, for each label, reference values plotted against test values. 
Accompanied by a histogram of the difference in values.

.. image:: 1to1_label_0.png
    :width: 300pt

.. image:: 1to1_label_1.png
    :width: 300pt

.. image:: 1to1_label_2.png
    :width: 300pt

Step 5: Model Spectra (``draw_spectra``, ``diagnostics``)
---------------------------------------------------------

Now that we have the model and labels for the test objects, ``The Cannon`` can
"draw" spectra for each test object.

    >>> from spectral_model import draw_spectra
    >>> cannon_set = draw_spectra(model, test_set)

We can now perform a final set of diagnostic checks.

    >>> from spectral_model import diagnostics
    >>> diagnostics(cannon_set, test_set, model)

The output of these diagnostics with sample plots are listed below.

5.1) A directory called SpectrumFits containing (for 10 randomly-selected stars) 
the Cannon fitted spectra overlaid with the 'true' (data) spectra, as well as 
the two compared in a 1-to-1 plot.

.. image:: Star500.png

5.2) For each label, the residuals of the spectra fits stacked and sorted by 
that label. If the functional form of the model is comprehensive enough, then 
this should look like noise and there should be no systematic structure.

.. image:: residuals_sorted_by_label_0.png

.. image:: residuals_sorted_by_label_1.png

.. image:: residuals_sorted_by_label_2.png

5.3) The autocorrelation of the mean spectral residual. If the functional form 
of the model is comprehensive enough, then this should be a delta function. 

.. image:: residuals_autocorr.png
