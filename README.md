# hoomd3_phosphorylation

Repository with code used in Zippo, E., Dormann, D., Speck, T. & Stelzl, L.L. Molecular simulations of enzymatic phosphorylation of
disordered proteins and their condensates, *bioRxiv* (2024), doi: https://doi.org/10.1101/2024.08.15.607948 .

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


### Usage

**How to run the code**:

Use the python script *main.py* to run the MD simulation in HOOMD3 using an HPS-based model with phosphorylation reactions.
You can specify the input parameters and the type of simulation you want through flag options:

  -c, --create_conf     
  
                        The code will run in the 'create_initial_configuration' mode. Only cubic box are available for creating the
                        initial configuration, the box can be resized during the simulation using the flag '-br'. Give only one
                        number in the input 'box' of the input file.
                        
  -m {HPS,HPS_cp,CALVADOS2}, --model {HPS,HPS_cp,CALVADOS2}
  
                        The code will run in simulation mode. The argument of this flag must be the name of the coarse-grained
                        model to use in the simulation: HPS for HPS model<sup>2</sup>, HPS_cp for HPS plus cation-pi enhanced interactions<sup>4</sup>, CALVADOS2 for CALVADOS2 model<sup>3</sup>.
                        It can not be used together with '-c'.
                        
  -i INFILE, --infile INFILE
  
                        Input file with simulation parameters, logging file name and parameters, system file name.
                        
  -r RESCALE, --rescale RESCALE
  
                        Scale down rigid body interaction by 'RESCALE' percentage<sup>5</sup>. To use also in 'create_initial_configuration' mode to
                        incude the rescaled rigid body types (value of argmuent not important in this case).
                        
  -br BOXRESIZE BOXRESIZE BOXRESIZE, --boxresize BOXRESIZE BOXRESIZE BOXRESIZE
                        The simulation will be used to resize the box from the initial configuration to the sizes given in the
                        argument. The argument should be a list with the side lengths (Lx, Ly, Lz).
                        
  --mode {relax,ness,nophospho}
                        Default 'relax', phosphorylation is active without exchange SER/SEP with the chemical bath. If 'ness' also exchange
                        step is added. If 'nophospho' phosphorylation and exchange are deactivated.

**Input file**

The input file has to be parsed to the script *main.py* through the flag '-i'. It contains the information on topology and parameters necessary to run the simulation.
A template of the input file can be found in the folder *input_stats/*, or an example of input file can found in the *example/* folder.
A correct input file must contain the following entries: 
- Simulation parameters:
  'production_dt' = time step in ps;
  'production_steps' = total number of steps;
  'production_T' = temperature for production run in K;
  'ionic' = ionic strength in mol/L (M);
  'box' = box side lengths Lx,Ly,Lz in nm (or just Lx for cubic boxes and for 'create_initial_configuration' mode);
  'start' = 0->new simulation, 1->restart simulation;
  'contact_dist' = distance in nm for contact with active site in phosphorylation step;
  'Dmu' = chemical potential bias for phosphorylation step in kJ/mol (mu_adp-mu_atp), 1 entry for every enzyme in simulation;
  'dt_try_change' = time interval in MD steps to try phosphorylation step;
  'seed' = seed for random number generator.
- Logging time interval: 'dt_dump' = time interval in MD steps to save trajectory file; 'dt_log' = time interval in MD steps to save log file; 'dt_backup' = time interval in MD steps to backup; 'dt_time' = time interval in MD steps to print out the timestep.
- Files: 'stat_file' = file with residue parameters definition; 'file_start' = file with starting configuration; 'logfile' = name of the output files (i.e. dump file will be '{logfile}_dump.gsd'); 'sysfile' = system file with definition of number of chains, rigid bodies, active sites and phosphosites (look at next section *System file*).
- Backend: 'dev' = GPU or CPU; 'logging' = logging level according to *logging* python standard library.

**System file**


<small>1) Zippo, E., Dormann, D., Speck, T. & Stelzl, L.L. Molecular simulations of enzymatic phosphorylation of
disordered proteins and their condensates, *bioRxiv* (2024), doi: https://doi.org/10.1101/2024.08.15.607948 </small>

<small>2) </small>

<small>3) </small>

