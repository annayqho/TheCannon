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

