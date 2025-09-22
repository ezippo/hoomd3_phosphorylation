#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import itertools

import numpy as np
import hoomd

class PrintTimestep(hoomd.custom.Action):
    """
    A custom HOOMD action to print the elapsed time and current simulation step.

    This class is used to monitor and display the progress of a simulation by
    printing the elapsed time since the start and the current simulation timestep.

    Args:
        t_start (float): The start time in seconds (usually obtained from time.time()).
        production_steps (int): The total number of production steps in the simulation.
    """
    def __init__(self, t_start, production_steps):
        """
        Initialize the PrintTimestep action.

        Args:
            t_start (float): The start time in seconds.
            production_steps (int): The total number of production steps in the simulation.
        """
        self._t_start = t_start  # Store the start time
        self._production_steps = production_steps  # Store the total production steps

    def act(self, timestep):
        """
        Print the elapsed time and the current simulation timestep.

        Args:
            timestep (int): The current simulation timestep.
        """
        # Calculate the current elapsed time
        current_time = time.time()
        current_time = current_time - self._t_start

        # Print the elapsed time and the current simulation step
        print(f"Elapsed time {current_time} | Step {timestep}/{self._production_steps}")


def macros_from_infile(infile):
    """
    Parse an input file to extract macro definitions and their associated values.

    Args:
        infile (str): Path to the input file containing macro definitions. The file
                      should have each macro on a new line with its values. Lines 
                      starting with '#' are treated as comments and ignored.

    Returns:
        dict: A dictionary where the keys are macro names and the values are either
              a single value or a list of values associated with that macro.
    """
    # Initialize an empty dictionary to store macro definitions
    macro_dict = {}
    
    # Open the input file for reading
    with open(infile, 'r') as fid:
        # Iterate over each line in the file
        for line in fid:
            # Skip lines that are comments (start with '#') or are empty
            if not line.startswith("#") and not line.isspace():
                
                # Split the line into a list of words
                line_list = np.array(line.rsplit())
                
                # Find the position of any comment within the line
                comment_pos = next((i for i, value in enumerate(line_list) if value.startswith("#")), None)
                
                if comment_pos is not None:
                    # If a comment is found, only consider values before the comment
                    if comment_pos > 2:
                        macro_dict[line_list[0]] = line_list[1:comment_pos]  # Store multiple values as a list
                    else:
                        macro_dict[line_list[0]] = line_list[1]  # Store a single value
                else:
                    # If no comment is found, store all values associated with the macro
                    if len(line_list) > 2:
                        macro_dict[line_list[0]] = line_list[1:]  # Store multiple values as a list
                    else:
                        macro_dict[line_list[0]] = line_list[1]  # Store a single value
    
    # Return the dictionary containing all macro definitions and their values
    return macro_dict
    

def aa_stats_from_file(filename):
    '''
    Parameters
    ----------
    filename : str
        name of stats file.

    Returns
    -------
    aa_dict : dicct
        dict('amino acid name':[mass, charge, sigma, lambda])
    '''
    aa_dict = {}
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]!='#':
                line_list = line.rsplit()
                aa_dict[line_list[0]] = np.loadtxt(line_list[1:], dtype=float)
    return aa_dict


def _deprecated_aa_stats_sequence(filename, aa_dict):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    aa_dict : dict
        dict('amino acid name':[mass, charge, sigma, lambda]).

    Returns
    -------
    chain_id : list
        List of a.a. id numbers of the sequence.
    chain_mass : list
        List of a.a. masses of the sequence.
    chain_charge : list
        List of a.a. charges of the sequence.
    chain_sigma : list
        List of a.a. radia of the sequence.
    chain_pos : list
        List of a.a. position tuple (x,y,z) of the sequence.
    '''
    chain_id = []
    chain_mass = []
    chain_charge = []
    chain_sigma = []
    chain_pos = []
    aa_keys = list(aa_dict.keys()) 
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                aa_name = line_list[3]
                chain_id.append(aa_keys.index(aa_name))
                chain_mass.append(aa_dict[aa_name][0])
                chain_charge.append(aa_dict[aa_name][1])
                chain_sigma.append(aa_dict[aa_name][2])
                chain_pos.append( (float(line_list[6]), float(line_list[7]), float(line_list[8])) )
    return chain_id, chain_mass, chain_charge, chain_sigma, chain_pos

def aa_stats_sequence(filename, aa_dict):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    aa_dict : dict
        dict('amino acid name':[mass, charge, sigma, lambda]).

    Returns
    -------
    chain_id : list
        List of a.a. id numbers of the sequence.
    chain_mass : list
        List of a.a. masses of the sequence.
    chain_charge : list
        List of a.a. charges of the sequence.
    chain_sigma : list
        List of a.a. radia of the sequence.
    chain_pos : list
        List of a.a. position tuple (x,y,z) of the sequence.
    '''
    aa_keys = list(aa_dict.keys()) 
    u = mda.Universe(filename)
    chain_resnames = u.atoms.resnames

    chain_id = [ aa_keys.index(aa_name) for aa_name in chain_resnames]
    chain_mass = [ aa_dict[aa_name][0] for aa_name in chain_resnames ]
    chain_charge = [ aa_dict[aa_name][1] for aa_name in chain_resnames ]
    chain_sigma = [ aa_dict[aa_name][2] for aa_name in chain_resnames ]
    chain_pos = u.atoms.positions
    
    return chain_id, chain_mass, chain_charge, chain_sigma, chain_pos


def _deprecated_chain_id_from_pdb(filename, aa_dict):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    aa_dict : dict
        dict('amino acid name':[mass, charge, sigma, lambda]).

    Returns
    -------
    chain_id : list
        List of a.a. id numbers of the sequence.
    '''
    chain_id = []
    aa_keys = list(aa_dict.keys()) 
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                aa_name = line_list[3]
                chain_id.append(aa_keys.index(aa_name))
    return chain_id

def chain_id_from_pdb(filename, aa_dict):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    aa_dict : dict
        dict('amino acid name':[mass, charge, sigma, lambda]).

    Returns
    -------
    chain_id : list
        List of a.a. id numbers of the sequence.
    '''
    aa_keys = list(aa_dict.keys()) 
    u = mda.Universe(filename)
    chain_resnames = u.atoms.resnames
    chain_id = [ aa_keys.index(aa_name) for aa_name in chain_resnames]
    
    return chain_id

def _deprecated_chain_positions_from_pdb(filename, relto=None, chain_mass=None, unit='nm'):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    relto : str (default None)
        if None: extracts raw positions from pdb file;
        if 'com': computes positions relative to the center-of-mass. In this case, it is mandatory to specify the parameter 'chain_mass';
        if 'cog': computes positions relative to the center-of-geometry.
    chain_mass : list (default None)
        if relto='com', you need to specify the list of a.a. masses of the sequence;
    unit : str (default 'nm')
        if 'nm': devides the postitions values by 10
        if 'A': keeps the values in Angstrom

    Returns
    -------
    chain_pos : ndarray
        Numpay array of a.a. positions (x,y,z) of the sequence.
    '''
    chain_pos_l = []
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                chain_pos_l.append( (float(line_list[6]), float(line_list[7]), float(line_list[8])) )
    if unit=='A':
        chain_pos = np.array(chain_pos_l)
    elif unit=='nm':
        chain_pos = np.array(chain_pos_l)/10.

    if relto==None:
        return chain_pos
    elif relto=='cog':
        return chain_pos - np.mean(chain_pos, axis=0)
    elif relto=='com':
        reshaped_mass = np.reshape( chain_mass, (len(chain_mass),1) )
        chain_com_pos = np.sum(chain_pos * reshaped_mass, axis=0) / np.sum(chain_mass)
        return chain_pos - chain_com_pos
    else:
        print("ERROR: relto option can only be None, 'cog' or 'com'. The insterted value is not valid! ")
        exit()

def chain_positions_from_pdb(filename, relto=None, chain_mass=None, unit='nm'):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    relto : str (default None)
        if None: extracts raw positions from pdb file;
        if 'com': computes positions relative to the center-of-mass. In this case, it is mandatory to specify the parameter 'chain_mass';
        if 'cog': computes positions relative to the center-of-geometry.
    chain_mass : list (default None)
        if relto='com', you need to specify the list of a.a. masses of the sequence;
    unit : str (default 'nm')
        if 'nm': devides the postitions values by 10
        if 'A': keeps the values in Angstrom

    Returns
    -------
    chain_pos : ndarray
        Numpay array of a.a. positions (x,y,z) of the sequence.
    '''
    u = mda.Universe('input_stats/com_UPF1.pdb')
    if unit=='A':
        chain_pos = u.atoms.positions
    elif unit=='nm':
        chain_pos = u.atoms.positions/10.

    if relto==None:
        return chain_pos
    elif relto=='cog':
        return chain_pos - np.mean(chain_pos, axis=0)
    elif relto=='com' and chain_mass is not None:
        reshaped_mass = np.reshape( chain_mass, (len(chain_mass),1) )
        chain_com_pos = np.sum(chain_pos * reshaped_mass, axis=0) / np.sum(chain_mass)
        return chain_pos - chain_com_pos
    else:
        print("ERROR: relto option can only be None, 'cog' or 'com'. The insterted value is not valid! ")
        exit()


def system_from_file(filename):
    '''
    Parameters
    ----------
    filename : str
        name of sysfile.

    Returns
    -------
    dict_list : list of dicts
        [ dict('mol1': ['pdb1', 'N1', 'rigid1', 'active_sites1', 'phospho_sites1']), 
          dict('mol2': ['pdb2', 'N2', 'rigid2', 'active_sites2', 'phospho_sites2']),
          ... ]
    '''
    dict_list = []
    with open(filename, 'r') as fid:
        for line in fid:
            mol_dict = dict()              
            if not line.startswith("#") and not line.isspace():
                line_list = np.array(line.rsplit())
                mol_dict['mol'] = line_list[0]    # mol name 
                mol_dict['pdb'] = line_list[1]    # pdb file 
                mol_dict['N'] = line_list[2]      # N molecules 
                mol_dict['rigid'] = line_list[3]    # rigid body indexes
                mol_dict['active_sites'] = line_list[4]    # active site indexes
                mol_dict['phospho_sites'] = line_list[5]    # phospho site indexes
                dict_list += [mol_dict]
                
    return dict_list


def read_rigid_indexes(rigid_str):
    """
    Parse a string defining rigid body indexes and return a list of numpy arrays
    representing these indexes.

    Args:
        rigid_str (str): A string representing rigid body indexes. The string can be '0'
                         to indicate no rigid bodies, or a comma-separated list of ranges
                         (e.g., '1-3,5-7') specifying the start and end of each rigid body.

    Returns:
        list: A list of numpy arrays, where each array contains the indexes of atoms
              belonging to a rigid body. If `rigid_str` is '0', an empty list is returned.
    """
    
    # Initialize an empty list to store the rigid body indexes
    rigid_list = []

    # If the input string is '0', return an empty list (no rigid bodies)
    if rigid_str == '0':
        return rigid_list
    else:
        # Split the string by commas to separate the rigid body ranges
        rigid_bodies = rigid_str.rsplit(',')
        
        # Iterate over each range of rigid bodies
        for body in rigid_bodies:
            # Split the range into its start and end indexes, converting them to integers
            init_body, end_body = np.array(body.rsplit('-'), dtype=int)
            
            # Create a numpy array representing the range of indexes for this rigid body
            # Adjusting to 0-based indexing (subtracting 1 from each index)
            rigid_list += [np.linspace(init_body - 1, end_body - 1, end_body - init_body + 1, endpoint=True, dtype=int)]
    
    # Return the list of numpy arrays representing the rigid body indexes
    return rigid_list
    

def rigidbodies_from_syslist(syslist, chain_lengths_l, aa_param_dict, rescale=0):
    """
    Generate rigid body configurations from a list of molecular systems.

    Args:
        syslist (list): A list of dictionaries where each dictionary contains information 
                        about a molecule, including PDB data and rigid body specifications.
        chain_lengths_l (list): A list of integers representing the length of each chain in 
                                the system.
        aa_param_dict (dict): A dictionary containing amino acid parameters. The keys are 
                              amino acid types, and the values are lists containing mass, charge
                              sigma and lambda.
        rescale (int, optional): If non-zero, the function uses rescaled amino acid types 
                                 for rigid body configurations. Default is 0 (no rescale).

    Returns:
        tuple: A tuple containing:
            - rigid (hoomd.md.constrain.Rigid): A HOOMD rigid body constraint object with 
                                                the defined rigid bodies.
            - rigid_masses_l (list): A list of masses for each rigid body.
            - n_rigids_l (list): A list of integers representing the number of rigid bodies 
                                 in each molecule.
            - R_type_list (list): A list of rigid body types generated.
    """
    # Number of molecules in the system
    n_mols = len(syslist)

    # List of amino acid types
    aa_type = list(aa_param_dict.keys())
    
    aa_mass = []
    aa_charge = []
    for k in aa_type:
        aa_mass.append(aa_param_dict[k][0])
        aa_charge.append(aa_param_dict[k][1])

    # If rescaling is enabled, generate rescaled amino acid types
    if rescale!=0:
        aa_type_r = [f"{name}_r" for name in aa_type]
                                                            
    prev_rigids = 0   # Counter for the number of rigid bodies generated
    rigid = hoomd.md.constrain.Rigid()
    rigid_masses_l = []     # List to store the mass of each rigid body
    n_rigids_l = []       # List to store the number of rigid bodies in each molecule
    R_type_list = []         # List to store the names of the rigid body types
    
    # Loop over each molecule in the system
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        chain_id = chain_id_from_pdb(mol_dict['pdb'], aa_param_dict)

        if mol_dict['rigid']!='0':
            chain_mass = [aa_mass[chain_id[i]] for i in range(chain_lengths_l[mol])]       # mass for each amino acid in the chain
            chain_charge = [aa_charge[chain_id[i]] for i in range(chain_lengths_l[mol])]         # charge for each amino acid in the chain
            chain_rel_pos = chain_positions_from_pdb(mol_dict['pdb'], relto='com', chain_mass=chain_mass)   # positions relative to c.o.m. 
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])       # Read the rigid body indexes
            n_rigids_l += [len(rigid_ind_l)]     # Update the list of number of rigid bodies
            
            # Loop over each rigid body defined in the molecule
            for nr in range(n_rigids_l[-1]):
                rigid_mass = [chain_mass[i] for i in rigid_ind_l[nr]]      # Get the mass of the amino acids in the rigid body
                rigid_masses_l += [np.sum(rigid_mass)]               # Update the list of total mass of the rigid bodies
                rigid_rel_pos = chain_rel_pos[rigid_ind_l[nr]]        # Get the relative positions of the amino acids in the rigid body
                
                # Calculate the center of mass position for the rigid body
                reshaped_rigid_mass = np.reshape( rigid_mass, (len(rigid_mass),1) )
                rigid_com_rel_pos = np.sum(rigid_rel_pos * reshaped_rigid_mass, axis=0) / np.sum(rigid_mass)       

                rigid_rel_pos = rigid_rel_pos-rigid_com_rel_pos       # Adjust positions to be relative to the center of mass of the rigid body
                
                prev_rigids += 1
                rig_name = 'R' + str( prev_rigids )
                R_type_list += [rig_name]   # Update list of names of the rigid body types

                # Define the rigid body in HOOMD, with or without rescaling
                if rescale==0:
                    rigid.body[rig_name] = {
                        "constituent_types": [aa_type[chain_id[i]] for i in rigid_ind_l[nr]],
                        "positions": rigid_rel_pos,
                        "orientations": [(1,0,0,0)]*len(rigid_ind_l[nr]),
                        "charges": [ chain_charge[i] for i in rigid_ind_l[nr] ],
                        "diameters": [0.0]*len(rigid_ind_l[nr])
                        }
                else:
                    rigid.body[rig_name] = {
                        "constituent_types": [aa_type_r[chain_id[i]] for i in rigid_ind_l[nr]],
                        "positions": rigid_rel_pos,
                        "orientations": [(1,0,0,0)]*len(rigid_ind_l[nr]),
                        "charges": [ chain_charge[i] for i in rigid_ind_l[nr] ],
                        "diameters": [0.0]*len(rigid_ind_l[nr])
                        }
        else:
            # If no rigid bodies are defined, append 0 to the rigid bodies count list
            n_rigids_l += [0]

    return rigid, rigid_masses_l, n_rigids_l, R_type_list


def rigid_dict_from_syslist(syslist):
    """
    Create a dictionary containing rigid body information for each molecule in the system.

    Args:
        syslist (list): A list of dictionaries where each dictionary contains information
                        about a molecule, including the PDB file path, the number of chains (N),
                        and the rigid body definition.

    Returns:
        dict: A dictionary where each key corresponds to a molecule identifier and the value is another
              dictionary with the following keys:
              - "n_rigids" (int): The number of rigid bodies in the molecule.
              - "rigid_lengths" (list): A list of integers representing the lengths of each rigid body.
              - "free_lengths" (list): A list of integers representing the lengths of free (non-rigid) segments
                                        between rigid bodies.
              - "n_chains" (int): The number of chains for the molecule as specified in the input.
    """
    n_mols = len(syslist)    # Number of molecules in the system
    mol_keys = [syslist[mol]['mol'] for mol in range(n_mols)]     # molecule identifiers (keys) from the system list
    rigid_dict = dict()

    # Loop over each molecule in the system
    for mol in range(n_mols):
        key = mol_keys[mol]
        mol_dict = syslist[mol]

        u = mda.Universe(mol_dict['pdb'])
        chain_length = u.atoms.n_atoms

        if mol_dict['rigid']=='0':
            # If no rigid bodies, store the information with zero rigid bodies and the entire chain as free
            rigid_dict[key] = {
                                "n_rigids": 0, 
                                "rigid_lengths": [], 
                                "free_lengths": [chain_length],
                                "n_chains": int(mol_dict['N'])
                                }
        else:
            # Parse the rigid body indexes
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
            n_rigids = len(rigid_ind_l)        # counts of rigid bodies
            rigid_lengths = [ len(rigid_ind_l[nr]) for nr in range(n_rigids) ]    # lengths of each rigid bodies

            # compute lengths of free (non-rigid) segments between rigid bodies
            free_lengths = [ rigid_ind_l[0][0] ]   # Length before the first rigid body    
            for nr in range(n_rigids-1):
                free_lengths += [ rigid_ind_l[nr+1][0] - rigid_ind_l[nr][-1] -1 ]
            free_lengths += [ chain_length-1 - rigid_ind_l[-1][-1] ]    # Length after the last rigid body
            # Store the calculated information in the dictionary
            rigid_dict[key] = {
                                "n_rigids": n_rigids, 
                                "rigid_lengths": rigid_lengths, 
                                "free_lengths": free_lengths,
                                "n_chains": int(mol_dict['N'])
                                }
    return rigid_dict


def reordering_index(syslist):
    """
    Generate a reordered index list for molecules based on their rigid body and free segment structure.
    System configuration is created following these indexing order criteria:
    1) oreder by molecule type following the order in syslist (from sysfile)
        within the molecule type:
        2) virtual rigid body particles in order of appearence in the chain and non-rigid segments in order of appearence
        3) repeat virtual rigid body particles and non-rigid segments for the number of chains of the same molecule type
        4) all the particles beloning to the rigid bodies in order of appearence along the chain
        5) repeat the particles beloning to the rigid bodies for the number of chains of the same molecule type

    The reordered list has the indexes of the particles in the system following these order criteria:
    1) oreder by molecule type following the order in syslist (from sysfile)
        within the molecule type:
        2) virtual rigid body particles first
        3) then follows the order of appearence along the chain
        4) repeat for the number of chains of the same molecule type

    Args:
        syslist (list): A list of dictionaries where each dictionary contains information 
                        about a molecule, including the PDB file path, the number of chains (N),
                        and the rigid body definition.

    Returns:
        list: A list of integers representing the reordered indices of the monomers in the system,
              considering both rigid bodies and free segments.
    """

    rigid_dict = rigid_dict_from_syslist(syslist)       # Generate the rigid body information dictionary

    # Initialize the reordered list and counters
    reordered_list = []
    n_prev_freeR = 0         # Counter for virtual rigid body particles and free (non-rigid) segments
    n_prev_rig = 0

    # Calculate how many particles are in system before the particles belonging to the rigid bodies (i.e. total virtual rigid body particles and free particles)
    for key in (mol['mol'] for mol in syslist):
        mol_info = rigid_dict[key]
        n_prev_rig += mol_info['n_rigids'] * mol_info['n_chains']          # total count of virtual rigid body particles
        n_prev_rig += np.sum(mol_info['free_lengths']) * mol_info['n_chains']      # total count of free particles

    # Loop over each molecule to construct the reordered list
    for mol in syslist:
        key = mol['mol']
        mol_info = rigid_dict[key]

        if mol_info['n_rigids']==0:
            # Case 1: The molecule has no rigid bodies
            free_length = mol_info['free_lengths'][0]        # list of lengths of free segments (only one segment with length of entire chain for Case 1)
            # Repeat the free segment indices for each chain
            reordered_list.extend(range(n_prev_freeR, n_prev_freeR + free_length * mol_info['n_chains']))
            n_prev_freeR += free_length * mol_info['n_chains']    # Update counter
        else:
            # Case 2: The molecule has rigid bodies
            free_lengths = mol_info['free_lengths']         # list of lengths of free segments 
            rigid_lengths = mol_info['rigid_lengths']       # list of lengths of rigid bodies 

            # Loop over the number of chains of the same species of molecule
            for _ in range(rigid_dict[key]['n_chains']):
                # Append indices for all the virtual rigid body patricles in molecule 'mol' and for the free segment before the first rigid body 
                total_rigid_and_initial_free = mol_info['n_rigids'] + free_lengths[0]
                reordered_list.extend(range(n_prev_freeR, n_prev_freeR + total_rigid_and_initial_free))
                n_prev_freeR += total_rigid_and_initial_free         # Update counter
    
                # Loop over the rigid bodies in the molecule 'mol'
                for rigid_length, next_free_length in zip(rigid_lengths, free_lengths[1:]):
                    # Add indices for the rigid body
                    reordered_list.extend(range(n_prev_rig, n_prev_rig + rigid_length))
                    n_prev_rig += rigid_length       # Update counters
                    # Add indices for the free segment after the rigid body
                    reordered_list.extend(range(n_prev_freeR, n_prev_freeR + next_free_length))
                    n_prev_freeR += next_free_length   # Update counters
                    
    return reordered_list



def protein_moment_inertia(chain_rel_pos, chain_mass, chain_sigma=None):
    '''
    Parameters
    ----------
    chain_rel_pos : list
        List of a.a. position tuple (x,y,z) of the sequence.
    chain_mass : list
        List of a.a. masses of the sequence.
    chain_sigma : list, optional
        List of a.a. radia of the sequence.

    Returns
    -------
    I : array
        Moment of inertia tensor.
    '''
    I = np.zeros((3,3))
    if chain_sigma==None:      # particle is a point
        for i,r in enumerate(chain_rel_pos):
            I += chain_mass[i]*( np.dot(r,r)*np.identity(3) - np.outer(r, r) )
    else:                      # particle is a sphere
        for i,r in enumerate(chain_rel_pos):
            I_ref = 2 / 5 * chain_mass[i]*chain_sigma[i]*chain_sigma[i]*np.identity(3)
            I += I_ref + ck1d_mass[i]*( np.dot(r,r)*np.identity(3) - np.outer(r, r) )
    return I
    

def U_ashbaugh_hatch(r, sigma, lambda_hps, epsilon=0.8368):
    '''
    HPS model: Ashbaugh-Hatch potential
    Lennard-Jones potential corrected for hydrophobicity effects
    
    Parameters
    ----------
    r : float
        Distance between particles (potential variable).
    sigma : float
        Lennard-Jones parameter (particle size).
    lambda_hps : float
        Hydrophobicity scale, between 0(hydrophilic) and 1(hydrophobic).
    epsilon : float, optional
        Lennard-Jones energy parameter. The default is 0.8368.

    Returns
    -------
    U : float
        U_ashbaugh(r).

    '''
    rmin = 2.**(1./6.) * sigma
    Ulj = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
    
    if r <= rmin:
        U = Ulj + (1-lambda_hps)*epsilon
    else:
        U = lambda_hps*Ulj
    
    return U

def F_ashbaugh_hatch(r, sigma, lambda_hps, epsilon=0.8368):
    '''
    HPS model: Ashbaugh-Hatch force
    Lennard-Jones force corrected for hydrophobicity effects
    
    F_ashbaugh = - 
    
    Parameters
    ----------
    r : float
        Distance between particles (potential variable).
    sigma : float
        Lennard-Jones parameter (particle size).
    lambda_hps : float
        Hydrophobicity scale, between 0(hydrophilic) and 1(hydrophobic).
    epsilon : float, optional
        Lennard-Jones energy parameter. The default is 0.8368.

    Returns
    -------
    F : float
        F_ashbaugh(r).

    '''
    rmin = 2.**(1./6.) * sigma
    Flj = 24*epsilon*( 2*(sigma/r)**12 - (sigma/r)**6 )/r
    
    if r <= rmin:
        F = Flj
    else:
        F = lambda_hps*Flj
    
    return F


def Ulist_ashbaugh(sigma, lambda_hps, r_max, r_min=0.4, n_bins=100, epsilon=0.8368):     
    '''
    HPS model:
    Table Ashbaugh-Hatch potential for HOOMD3 simulations

    Parameters
    ----------
    sigma : float, tuple, list, narray
        Lennard-Jones parameter: average particle size or sizes of the 2 involved particles.
    lambda_hps : flaot, tuple, list, ndarray
        HPS parameter: average hydrophobicity scale or hydrophobicity scales of the 2 involved particles.
    r_max : float
        Maximum potential variable.
    r_min : float, optional
        Minimum potential variable. The default is 0.
    n_bins : int, optional
        Size of the tble potential. The default is 100.
    epsilon : float, optional
        Lennard-Jones parameter: energy scale. The default is 0.8368.

    Returns
    -------
    Ulist : list
        List of U_ashbaugh(r) with r in range r_min to r_max in n_bins steps.

    '''
    # Error handlig for bad sigma values
    if type(sigma)==float:
        s = sigma        
    else:
        try:
            if len(sigma)==2:
                s = (sigma[0] + sigma[1])/2.
            else:
                raise IndexError('sigma and lambda_hps params in Ulist_ashbaugh must be float or iterable with 2 values!')
        except:
            print('TypeError: sigma and lambda_hps params in Ulist_ashbaugh must be float or iterable with 2 values!')
    
    # Error handlig for bad lambda_hps values
    if type(lambda_hps)==float:
        l_hps = lambda_hps 
    else:
        try:
            if len(lambda_hps)==2:
                l_hps = (lambda_hps[0] + lambda_hps[1])/2.
            else:
                raise IndexError('sigma and lambda_hps params in Ulist_ashbaugh must be float or iterable with 2 values!')
        except:
            print('TypeError: sigma and lambda_hps params in Ulist_ashbaugh must be float or iterable with 2 values!')
    
    r_range = np.linspace(r_min, r_max, n_bins, endpoint=False)
    Ulist = [ U_ashbaugh_hatch(r, s, l_hps, epsilon) for r in r_range ]
    
    return Ulist
    

def Flist_ashbaugh(sigma, lambda_hps, r_max, r_min=0.4, n_bins=100, epsilon=0.8368):
    '''
    HPS model:
    Table Ashbaugh-Hatch force for HOOMD3 simulations
    
    F_ashbaugh = - d U_ashbaugh / d r

    Parameters
    ----------
    sigma : float, tuple, list, narray
        Lennard-Jones parameter: average particle size or sizes of the 2 involved particles.
    lambda_hps : flaot, tuple, list, ndarray
        HPS parameter: average hydrophobicity scale or hydrophobicity scales of the 2 involved particles.
    r_max : float
        Maximum potential variable.
    r_min : float, optional
        Minimum potential variable. The default is 0.
    n_bins : int, optional
        Size of the tble potential. The default is 100.
    epsilon : float, optional
        Lennard-Jones parameter: energy scale. The default is 0.8368.

    Returns
    -------
    Flist : list
        List of F_ashbaugh(r) with r in range r_min to r_max in n_bins steps.

    '''
    # Error handlig for bad sigma values
    if type(sigma)==float:
        s = sigma        
    else:
        try:
            if len(sigma)==2:
                s = (sigma[0] + sigma[1])/2.
            else:
                raise IndexError('sigma and lambda_hps params in Flist_ashbaugh must be float or iterable with 2 values!')
        except:
            print('TypeError: sigma and lambda_hps params in Flist_ashbaugh must be float or iterable with 2 values!')
    
    # Error handlig for bad lambda_hps values
    if type(lambda_hps)==float:
        l_hps = lambda_hps 
    else:
        try:
            if len(lambda_hps)==2:
                l_hps = (lambda_hps[0] + lambda_hps[1])/2.
            else:
                raise IndexError('sigma and lambda_hps params in Flist_ashbaugh must be float or iterable with 2 values!')
        except:
            print('TypeError: sigma and lambda_hps params in Flist_ashbaugh must be float or iterable with 2 values!')
    
    r_range = np.linspace(r_min, r_max, n_bins, endpoint=False)
    Flist = [ F_ashbaugh_hatch(r, s, l_hps, epsilon) for r in r_range ]
    
    return Flist



def compute_yukawa_params(tempK, ionic):
    """
    Compute the Yukawa potential parameters for a given temperature and ionic strength.

    Args:
        tempK (float): The temperature in Kelvin.
        ionic (float): The ionic strength of the solution in mol/L.

    Returns:
        tuple:
            yukawa_eps (float): The Yukawa potential depth, representing the strength of the Yukawa interaction (in kJ/mol).
            yukawa_kappa (float): The inverse Debye length, representing the screening effect of the ionic solution (in 1/nm).
    """
    # Calculate the thermal energy in kJ/mol
    RT = 8.3145 * tempK * 1e-3

    # Define a lambda function to calculate the dielectric constant of water (epsw)
    fepsw = lambda T: 5321/T + 233.76 - 0.9297*T + 0.1417*1e-2*T*T - 0.8292*1e-6*T**3

    # Calculate the dielectric constant of water at the given temperature
    epsw = fepsw(tempK)

    # Calculate the Bjerrum length (lB)
    lB = 1.6021766**2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / RT

    # Calculate the inverse Debye screening length (yukawa_kappa)
    yukawa_kappa = np.sqrt(8 * np.pi * lB * ionic * 6.022 / 10)

    # Calculate the Yukawa potential depth (yukawa_eps)
    yukawa_eps = lB * RT

    # Return the Yukawa potential depth and the inverse Debye length as a tuple
    return yukawa_eps, yukawa_kappa


def compute_distances_pbc(p1, p2, x_side, y_side, z_side):
    """
    Compute the distances between two arrays of particles in a box
    with periodic boundary conditions.

    Args:
        p1 (numpy array): Positions of particles in group 1.
                          Shape: (N1, 3), where N1 is the number of particles.
        p2 (numpy array): Positions of particles in group 2.
                          Shape: (N2, 3), where N2 is the number of particles.
        x_side (float):   Size of the side in x direction.
        y_side (float):   Size of the side in y direction.
        z_side (float):   Size of the side in z direction.

    Returns:
        numpy array: Distances between the particles.
                     Shape: (N1, N2), where N1 and N2 are the number of particles in the two groups.
    """
    # Compute the minimum image distance in each coordinate
    dx = np.abs(p1[:, 0, None] - p2[:, 0])
    dy = np.abs(p1[:, 1, None] - p2[:, 1])
    dz = np.abs(p1[:, 2, None] - p2[:, 2])

    # Apply periodic boundary conditions
    dx = np.minimum(dx, x_side - dx)
    dy = np.minimum(dy, y_side - dy)
    dz = np.minimum(dz, z_side - dz)

    # Compute the Euclidean distance
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    return dist


def compute_single_distance_pbc(p1, p2, box_size):
    """
    Compute the distances between two arrays of particles in a cubic box
    with periodic boundary conditions.

    Args:
        p1 (numpy array): Positions of particle 1.
                          Shape: (3)
        p2 (numpy array): Positions of particle 2.
                          Shape: (3)
        box_size (float): Size of the cubic box.

    Returns:
        float: Distance between the particles.
    """
    # Compute the minimum image distance in each coordinate
    dx = np.abs(p1[0] - p2[0])
    dy = np.abs(p1[1] - p2[1])
    dz = np.abs(p1[2] - p2[2])

    # Apply periodic boundary conditions
    dx = np.minimum(dx, box_size - dx)
    dy = np.minimum(dy, box_size - dy)
    dz = np.minimum(dz, box_size - dz)

    # Compute the Euclidean distance
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    return dist


def compute_center(pos):
    n = len(pos)
    center_pos = np.array([ np.sum(pos[:,i])/n for i in range(3) ])
    return center_pos


def generate_positions_cubic_lattice(n_chains, box_length):
    """Generate random positions for the chains in a cubic lattice in the simulation box."""
    sites_per_side = math.ceil(n_chains**(1/3))
    spacing = box_length / sites_per_side
    x = np.linspace(-box_length / 2, box_length / 2, sites_per_side, endpoint=False)
    positions = np.array(list(itertools.product(x, repeat=3))) + [spacing / 2] * 3
    np.random.shuffle(positions)
    return positions[:n_chains]





if __name__=='__main__':
    print('hello')
