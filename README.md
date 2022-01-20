# Multimodal Surface Matching for python

This package is a python wrapper for Multimodal Surface Matching (MSM),
a mesh alignment method from the Connectome Workbench.
It turns MSM into a scikit_learn compatible model.

It allows to train, test and evaluate transformations between a source
and a target dataset, which in our use case consists of
fMRI data on the human cortical surface.

## Requirements

[FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
should be installed and available from `$PATH` (if usure, run `which fsl`
to check for local installation).

## Install

In a dedicated `conda` env:

```
pip install -r requirements.txt
pip install -e .
```

## Dev install

On top of running the usual install commands, install dev dependencies with:

```
pip install -r requirements-dev.txt
```

This will allow to run tests locally:

```
pytest
```
