# hoomd3_phosphorylation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15207491.svg)](https://doi.org/10.5281/zenodo.15207491)

Repository with code used in Zippo, E., Dormann, D., Speck, T. & Stelzl, L.L. Molecular simulations of enzymatic phosphorylation of
disordered proteins and their condensates, *Nat.Commun.* (2025), doi: https://doi.org/10.1038/s41467-025-59676-4 .

Supporting data can be found on the Zenodo platform (Zippo, E. (2024). Supporting Data: Molecular simulations of enzymatic phosphorylation of disordered proteins and their condensates. Zenodo. https://doi.org/10.5281/zenodo.13833526).

It contains the framework to easily run coarse-grained Molecular Dynamics (MD) simulations of multi-domain proteins with phosphorylation reaction Monte Carlo steps using HPS-derived models<sup>1</sup>.

## Overview
- **src/hps_phosphorylation/**: *hps_phosphorylation* code to run MD simulation with phosphorylation reaction.
- **example/**: folder with examples of usage of *hps_phosphorylation*.
- **input_stats/**: folder with pdb files, amino-acids parameters from HPS<sup>2</sup> model and CALVADOS2<sup>3</sup>, input file template
- **main.py**: python script to run the simulations with *hps_phosphorylation*.
- **paper/**: pdf of the paper<sup>1</sup> and jupyter notebooks for analysis and plotting used in the paper.

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
The installation should take only few seconds.

**HOOMD installation tips**

In order to be able to install the custom *ashbaugh_plugin*, necessary to run the code, it is suggested to build HOOMD-blue from source.
Information can be found here at *https://hoomd-blue.readthedocs.io/en/v3.8.1/building.html* . It is sugegsted to build it in a virtual environment with all the prerequisites.
It is common to run into trouble with the recognition of packages paths and with packages compatibilities. 
If this is the case, most problems can be solved by providing the path to cmake through a cmake flag or by defining the appropriate global environment variable.
The main installation steps are:
  - create virtual environment and activate it
  - install prerequisites
  - **git clone --branch v3.8.1 --recursive https://github.com/glotzerlab/hoomd-blue**
  - **cd hoomd-blue**
  - **mkdir build**
  - **cd build**
  - when using a virtual environment, set the cmake prefix path before running cmake: **export CMAKE_PREFIX_PATH=\<path-to-environment\>**
  - **cmake ../ -D\<flags-name\>=\<value\>**
  - **make install**

Mandatory flags for the cmake are: **-DENABLE_GPU=on** to enable GPU computations, **-DSINGLE_PRECISION=on** for faster simulations if not interested in double precision computations, 
Optional flags: **-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native** to optimize the build for your processor, **-DPYTHON_EXECUTABLE=/path/to/python3 -DCMAKE_CUDA_COMPILER=/location/of/nvcc/or/hipcc -D\<not-found-package-name\>_ROOT=/path/to/not/found/package** if not automatically detected by cmake, **-DCMAKE_INSTALL_PREFIX=/path/to/install/hoomd** if you want to install hoomd in a specific location.


### Usage

**How to run the code**:

Use the python script *main.py* to run the MD simulation in HOOMD3 using an HPS-based model with phosphorylation reactions.
You can specify the input parameters and the type of simulation you want through flag options:

  -c, --create_conf  :  The code will run in the 'create_initial_configuration' mode. Only cubic box are available for creating the
                        initial configuration, the box can be resized during the simulation using the flag '-br'. Give only one
                        number in the input 'box' of the input file.
                        
  -m {HPS,CALVADOS}, --model {HPS,CALVADOS}  :  
                        The code will run in simulation mode. The argument of this flag must be the name of the coarse-grained
                        model to use in the simulation: HPS for HPS model<sup>2</sup> or HPS plus cation-pi enhanced interactions<sup>3</sup>, CALVADOS for CALVADOS models (1,2 or 3) <sup>4,5,6</sup>.
                        It can not be used together with '-c'.
                        
  -i INFILE, --infile INFILE  :  Input file with simulation parameters, logging file name and parameters, system file name.
                        
  -r RESCALE, --rescale RESCALE  :    
                        Scale down rigid body interaction by 'RESCALE' percentage<sup>7</sup>. To use also in 'create_initial_configuration' mode to
                        incude the rescaled rigid body types (value of argmuent not important in this case).
                        
  -br BOXRESIZE BOXRESIZE BOXRESIZE, --boxresize BOXRESIZE BOXRESIZE BOXRESIZE  :  
                        The simulation will be used to resize the box from the initial configuration to the sizes given in the
                        argument. The argument should be a list with the side lengths (Lx, Ly, Lz).

  -cp, --cationpi  :  If specified, an additional Lennard-Jones pair potential will be added between cationic and aromatic residues.

  -n, NETWORK, --network NETWORK :
                        The folded domains will be fixed using elastic network instead of rigid bodies, as modelled in CALVADOS3. 
                        Give in input NETWORK the name of the file in which the network distances will be written in 'create_initial_configuration' mode,
                        or in which the network distances have to be read from in simulation mode.

                        
  --mode {relax,ness,nophospho}  :  
                        Default 'relax', phosphorylation is active without exchange SER/SEP with the chemical bath. If 'ness' also exchange
                        step is added. If 'nophospho' phosphorylation and exchange are deactivated.

  --logenergy    :   If specified, the log file will store also the potential energy acting on each particle for each pair potential.
                        

**Input file**

The input file has to be parsed to the script *main.py* through the flag '-i'. It contains the information on topology and parameters necessary to run the simulation.
A template of the input file can be found in the folder *input_stats/*, or an example of input file can be found in the *example/* folder.
A correct input file must contain the following entries: 
1. Simulation parameters:
  - 'production_dt' = time step in ps;
  - 'production_steps' = total number of steps;
  - 'production_T' = temperature for production run in K;
  - 'ionic' = ionic strength in mol/L (M);
  - 'box' = box side lengths Lx,Ly,Lz in nm (or just Lx for cubic boxes and for 'create_initial_configuration' mode);
  - 'start' = 0->new simulation, 1->restart simulation;
  - 'contact_dist' = distance in nm for contact with active site in phosphorylation step;
  - 'bath_dist' = distance in nm for considering enzyme and substrate far enough for the reservoir exchange step. Necessary in simulation mode "ness";
  - 'Dmu' = chemical potential bias for phosphorylation step in kJ/mol (mu_adp-mu_atp), 1 entry for every enzyme in simulation;
  - 'dt_try_change' = time interval in MD steps to try phosphorylation step;
  - 'dt_bath' = time interval in MD steps to try reservoir exchange step. Necessary in simulation mode "ness";
  - 'seed' = seed for random number generator.
2. Logging time interval:
  - 'dt_dump' = time interval in MD steps to save trajectory file;
  - 'dt_log' = time interval in MD steps to save log file;
  - 'dt_backup' = time interval in MD steps to backup;
  - 'dt_time' = time interval in MD steps to print out the timestep.
3. Files:
  - 'stat_file' = file with residue parameters definition;
  - 'file_start' = file with starting configuration;
  - 'logfile' = name of the output files (i.e. dump file will be '{logfile}_dump.gsd');
  - 'sysfile' = system file with definition of number of chains, rigid bodies, active sites and phosphosites (look at next section *System file*).
4. Backend:
  - 'dev' = GPU or CPU;
  - 'logging' = logging level according to *logging* python standard library.

**System file**

The system file contains information about the topology of the system and its path has to be included in the input file under the entry 'sysfile'.
A template of the system file can be found in the folder *input_stats/*, or an example can be found in the *example/* folder.
The system file is structure in the following way:

1. its a table in which each line contains info about one species of molecules.
2. within a line, the entries are:
  - 'mol': nickname of reference for the molecule species (e.g. TDP43)
  - 'pdb': path to pdb file of the molecule species, needed for parsing sequence and positions in 'create_initial_configuration' mode
  - 'N': number of chains of the same molecule species in the simulation box
  - 'rigid': specify the portion of the molecule to keep fixed as a rigid body.
            Use '0' for no-rigid bodies; specify a range from i-th to j-th residue of the sequence to keep rigid using the format 'i-j';
            in case of multiple rigid bodies in the molecule, separate the ranges with a comma ',' (e.g. '3-80,102-140' are two rigid bodies, one from residue 3 to 80 and another from 102 to 140).
  - 'active_sites': specify the residues to be considered active site when computing the distances in the phosphorylation step.
            Use '0' for no active sites; specify active site residues as a list of serial numbers separated by comma (e.g. '149,150,151' means that the active site residues are residue 149,150 and 151).
  - 'phospho_sites': specify the phosphosites (residues that can be phosphorylated) for the phosphorylation step.
            Use '0' for no phosphosites; specify phosphosites residues as a list of serial numbers separated by comma (e.g. '6,147,150' means that residue 6,147 and 150 can be phosphorylated);
            use 'SER' to select all the serines of the molecule; use 'SER:i-j' to select all the serines in the sequence portion between residue 'i' and residue 'j'.



<small>1) Zippo, E., Dormann, D., Speck, T. & Stelzl, L.L. Molecular simulations of enzymatic phosphorylation of
disordered proteins and their condensates, *Nat.Commun.* (2025), doi: https://doi.org/10.1038/s41467-025-59676-4 </small>

<small>2) Dignon, G. L., Zheng, W., Kim, Y. C., Best, R. B. & Mittal, J. Sequence determinants of
protein phase behavior from a coarse-grained model, *PLoS Comput. Biol.* 14, e1005941 (2018) </small>

<small>3) Tejedor, A. R., Garaizar, A., Ramírez, J. & Espinosa, J. R. ‘RNA modulation of transport
properties and stability in phase-separated condensates, *Biophysical Journal* 120, 5169–5186 (2021) </small>

<small>4) Tesei G., Schulze T. K., Crehuet R., Lindorff-Larsen K.  Accurate model of liquid-liquid phase behavior of intrinsically disordered proteins from optimization of single-chain properties, *PNAS* (2021), 118(44):e2111696118 </small>

<small>5) Tesei G., Lindorff-Larsen K.  Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range, *Open Research Europe* (2022), 2(94). </small>

<small>6) Cao F., von Bülow S., Tesei G., Lindorff-Larsen K.  A coarse-grained model for disordered and multi-domain proteins, *Protein Science* (2024), 33(11):e5172 </small>

<small>7)  Krainer, G. et al. Reentrant liquid condensate phase of proteins is stabilized by hydrophobic
and non-ionic interactions, *Biophysical Journal* 120, 28a (2021) </small>
