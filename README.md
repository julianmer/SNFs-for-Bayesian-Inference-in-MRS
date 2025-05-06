# Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy

[Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy]()

## Abstract

Magnetic resonance spectroscopy (MRS) is a non-invasive technique to measure the metabolic composition of tissues, offering valuable insights into neurological disorders, tumor detection, and other metabolic dysfunctions. However, accurate metabolite quantification is hindered by challenges such as spectral overlap, low signal-to-noise ratio, and various artifacts. Traditional methods like linear-combination modeling are susceptible to ambiguities and commonly only provide a theoretical lower bound on estimation accuracy in the form of the Cram´er-Rao bound. This work introduces a Bayesian inference framework using Sylvester normalizing flows (SNFs) to approximate posterior distributions over metabolite concentrations, enhancing quantification reliability. A physics-based decoder incorporates prior knowledge of MRS signal formation, ensuring realistic distribution representations. We validate the method on simulated 7T proton MRS data, demonstrating accurate metabolite quantification, well-calibrated uncertainties, and insights into parameter correlations and multi-modal distributions.

## Overview

This repository consists of the following Python scripts:
* The `train.py` implements the pipeline to train (and test) the deep learning approaches.
* The `sweep.py` defines ranges to sweep for optimal hyperparamters using Weights & Biases.
* The `frameworks/` folder holds the frameworks for model-based and data-driven methods.
  * The `framework.py` defines the framework class to inherit from.
  * The `frameworkFSL.py` consists of a wrapper for FLS-MRS for easy use.
  * The `frameworkLCM.py` is a wrapper for LCModel.
  * The `frameworkNN.py` holds the base framework class for deep learning-based methods. 
  * The `frameworkSNF.py`contains the framework class for Sylvester normalizing flows.
  * The `lcmodel/` dictionary holds the LCModel binaries and executables.
* The `loading/` folder holds the scripts to automate the loading of MRS data formats.
  * The `dicom.py` defines functions for the DICOM loader (Philips).
  * The `lcmodel.py` contains loaders for the LCModel formats.
  * The `loadBasis.py` holds the loader for numerous basis set file formats.
  * The `loadConc.py` enables loading of concentration files provided by fitting software.
  * The `loadData.py` defines the loader for the MRS data.
  * The `loadLABRAW.py` allows to load the Philips lab raw format.
  * The `loadMRSI.py` contains functions for MRSI data.
  * The `philips.py` holds functions to load Philips data.
* The `models/` folder holds the models for the deep learning approaches.
  * The `flowModels.py` defines the flow models.
  * The `nnModels.py` defines the neural network models.
* The `simulation/` folder holds the scripts to simulate MRS spectra.
  * The `basis.py` has the basis set class to hold the spectra.
  * The `dataModules.py` are creating datasets by loading in-vivo data or simulating ad-hoc during training.
  * The `sigModels.py` defines signal models to simulate MRS spectra.
  * The `simulation.py` draws simulation parameters from distibutions to allow simulation with the signal model.
  * The `simulationDefs.py` holds predefined simulation parameters ranges.
* The `tests/` folder holds the test scripts.
  * The `test.py` defines the test class to inherit from.
  * The `testSNF.py` contains the test class for the Sylvester normalizing flow.
* The `utils/` folder holds helpful functionalities.
  * The `auxiliary.py` defines some helpful functions.
  * The `components.py` consists of functions to create signal components.
  * The `gpu_config.py` is used for the GPU configuration.
  * The `processing.py` defines functions for processing MRS data.
  * The `structures.py` implements helpful structures.

## Requirements

| Module            | Version |
|:------------------|:------:|
| fsl_mrs           | 2.1.20 |
| h5py              |  3.9.0 |
| matplotlib        |  3.8.2 |
| numpy             |  2.2.5 |
| pandas            |  2.2.3 |
| pydicom           |  3.0.1 |
| pytorch_lightning |  2.1.2 |
| scipy             | 1.15.2 |
| seaborn	    | 0.13.2 |
| shutup            |  0.2.0 |
| spec2nii          |  0.7.4 |
| torch             |  2.1.2 |
| torchmetrics      |  1.2.1 |
| tqdm              | 4.64.1 |
| wandb             | 0.16.1 |
