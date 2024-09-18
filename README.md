# hoomd3_phosphorylation

Repository with code ude in Zippo, E., Dormann, D., Speck, T. & Stelzl, L.L. Molecular simulations of enzymatic phosphorylation of
disordered proteins and their condensates, *bioRxiv* (2024), doi: https://doi.org/10.1101/2024.08.15.607948

It contains the framework to easily run coarse-grained Molecular Dynamics (MD) simulations of multi-domain proteins with phosphorylation reaction Monte Carlo steps using HPS-derived models^1.

## Overview
- **src/hps_phosphorylation/**: *hps_phosphorylation* code to run MD simulation with phosphorylation reaction.
- **example/**: folder with examples of usage of *hps_phosphorylation*.
- **input_stats/**: folder with pdb files, amino-acids parameters from HPS^2 model and CALVADOS2^3, ...
- **main.py**: python script to run the simulations with *hps_phosphorylation*.
- **paper/**: pdf of the paper^1 and jupyter notebooks for analysis and plotting used in the paper.

## *hps_phosphorylation*

### Installation

**Prerequisites**:
-  numpy
-  HOOMD-blue: code written for hoomd v3, tested with hoomd v3.8.1
-  gsd: code written for gsd v2, tested with gsd v2.8.1
-  ashbaugh_plugin: plugin for hoomd v3 with Ashbaugh-Hatch pair potential, can be found at *https://github.com/ezippo/ashbaugh_plugin*

**Installation 1** (suggested)
The code can be used by simply downloading the folder *hps_phosphorylation* in *src/* and appending the path to the package in your PYTHONPATH.
Copy the python script *main.py* to run the code.

**Installation 2**
The code is also available in PyPI-test and it can be installed by using the following command:
  pip install -i https://test.pypi.org/simple/ hps-phosphorylation
Copy the python script *main.py* to run the code.


### Usage

