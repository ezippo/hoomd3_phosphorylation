#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import gsd.hoomd
import hoomd

class PrintTimestep(hoomd.custom.Action):

    def __init__(self, t_start, production_steps):
        self._t_start = t_start
        self._production_steps = production_steps

    def act(self, timestep):
        current_time = time.time()
        current_time = current_time - self._t_start
        print(f"Elapsed time {current_time} | Step {timestep}/{self._production_steps} " )


def macros_from_infile(infile):
    '''
    Parameters
    ----------
    infile : str
        name of macros input file.

    Returns
    -------
    macro_dict : dict
        dict('macro name':macro value)
    '''
    macro_dict = {}
    with open(infile, 'r') as fid:
        for line in fid:
            if not line.startswith("#") and not line.isspace():
                line_list = np.array(line.rsplit())
                comment_pos = next((i for i, value in enumerate(line_list) if value.startswith("#")), None)
                if comment_pos is not None:
                    if comment_pos > 2:
                        macro_dict[line_list[0]] = line_list[1:comment_pos]
                    else:
                        macro_dict[line_list[0]] = line_list[1]
                else:
                    if len(line_list)>2:
                        macro_dict[line_list[0]] = line_list[1:] 
                    else:
                        macro_dict[line_list[0]] = line_list[1]
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
    chain_id = []
    aa_keys = list(aa_dict.keys()) 
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                aa_name = line_list[3]
                chain_id.append(aa_keys.index(aa_name))
    return chain_id


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
    rigid_list = []
    if rigid_str=='0':
        return rigid_list
    else:
        rigid_bodies = rigid_str.rsplit(',')
        for body in rigid_bodies:
            init_body, end_body = np.array(body.rsplit('-'), dtype=int)
            rigid_list += [ np.linspace(init_body-1, end_body-1, end_body-init_body+1, endpoint=True, dtype=int) ]
        return rigid_list
    

def rigidbodies_from_syslist(syslist, chain_lengths_l, aa_param_dict, rescale=0):
    
    n_mols = len(syslist)
    aa_type = list(aa_param_dict.keys())
    aa_mass = []
    aa_charge = []
    for k in aa_type:
        aa_mass.append(aa_param_dict[k][0])
        aa_charge.append(aa_param_dict[k][1])
    if rescale!=0:
        aa_type_r = [f"{name}_r" for name in aa_type]
                                                            
    prev_rigids = 0
    rigid = hoomd.md.constrain.Rigid()
    rigid_masses_l = []
    n_rigids_l = []
    R_type_list = []
    
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        chain_id = chain_id_from_pdb(mol_dict['pdb'], aa_param_dict)
        
        if mol_dict['rigid']!='0':
            chain_mass = [aa_mass[chain_id[i]] for i in range(chain_lengths_l[mol])]
            chain_charge = [aa_charge[chain_id[i]] for i in range(chain_lengths_l[mol])]
            chain_rel_pos = chain_positions_from_pdb(mol_dict['pdb'], relto='com', chain_mass=chain_mass)   # positions relative to c.o.m. 
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
            n_rigids_l += [len(rigid_ind_l)]
            
            for nr in range(n_rigids_l[-1]):
                rigid_mass = [chain_mass[i] for i in rigid_ind_l[nr]]
                rigid_masses_l += [np.sum(rigid_mass)]
                rigid_rel_pos = chain_rel_pos[rigid_ind_l[nr]] 
                reshaped_rigid_mass = np.reshape( rigid_mass, (len(rigid_mass),1) )
                rigid_com_rel_pos = np.sum(rigid_rel_pos * reshaped_rigid_mass, axis=0) / np.sum(rigid_mass)       # c.o.m. relative to the center of the molecule
                rigid_rel_pos = rigid_rel_pos-rigid_com_rel_pos             # positions of monomers of the rigid body relative to the c.o.m.
                prev_rigids += 1
                rig_name = 'R' + str( prev_rigids )
                R_type_list += [rig_name]
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
            n_rigids_l += [0]

    return rigid, rigid_masses_l, n_rigids_l, R_type_list


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


def rigid_dict_from_syslist(syslist):
    n_mols = len(syslist)
    mol_keys = [syslist[mol]['mol'] for mol in range(n_mols)]
    rigid_dict = dict()
    for mol in range(n_mols):
        key = mol_keys[mol]
        mol_dict = syslist[mol]
        chain_length = 0
        with open(mol_dict['pdb'], 'r') as fid:
            for line in fid:
                if line[0]=='A':
                    chain_length += 1

        if mol_dict['rigid']=='0':
            rigid_dict[key] = {
                                "n_rigids": 0, 
                                "rigid_lengths": [], 
                                "free_lengths": [chain_length],
                                "n_chains": int(mol_dict['N'])
                                }
        else:
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
            n_rigids = len(rigid_ind_l)
            rigid_lengths = [ len(rigid_ind_l[nr]) for nr in range(n_rigids) ]
            free_lengths = [ rigid_ind_l[0][0] ]
            for nr in range(n_rigids-1):
                free_lengths += [ rigid_ind_l[nr+1][0] - rigid_ind_l[nr][-1] -1 ]
            free_lengths += [ chain_length-1 - rigid_ind_l[-1][-1] ]
            rigid_dict[key] = {
                                "n_rigids": n_rigids, 
                                "rigid_lengths": rigid_lengths, 
                                "free_lengths": free_lengths,
                                "n_chains": int(mol_dict['N'])
                                }
    return rigid_dict


def reordering_index(syslist):
    n_mols = len(syslist)
    mol_keys = [syslist[mol]['mol'] for mol in range(n_mols)]
    rigid_dict = rigid_dict_from_syslist(syslist)

    reordered_list = []
    n_prev_freeR = 0
    n_prev_rig = np.sum([ rigid_dict[key]['n_rigids']*rigid_dict[key]['n_chains'] for key in mol_keys ])
    n_prev_rig += np.sum([ np.sum(rigid_dict[key]['free_lengths'])*rigid_dict[key]['n_chains'] for key in mol_keys ])
    for mol in range(n_mols):
        key = mol_keys[mol]
        if rigid_dict[key]['n_rigids']==0:
            tmp_length = rigid_dict[key]['free_lengths']
            for ch in range(rigid_dict[key]['n_chains']):
                reordered_list += [n_prev_freeR+i for i in range(tmp_length[0])]
                n_prev_freeR += tmp_length[0]
        else:
            tmp_length_free = rigid_dict[key]['free_lengths']
            tmp_length_rig = rigid_dict[key]['rigid_lengths']
            tmp_reord_list = []
            for ch in range(rigid_dict[key]['n_chains']):
                tmp_reord_list += [n_prev_freeR+i for i in range(rigid_dict[key]['n_rigids']+tmp_length_free[0])]
                n_prev_freeR += rigid_dict[key]['n_rigids']+tmp_length_free[0]
                for nr in range(rigid_dict[key]['n_rigids']):
                    tmp_reord_list += [n_prev_rig+i for i in range(tmp_length_rig[nr])]
                    tmp_reord_list += [n_prev_freeR+i for i in range(tmp_length_free[nr+1])]
                    n_prev_freeR += tmp_length_free[nr+1]
                    n_prev_rig += tmp_length_rig[nr]
            reordered_list += tmp_reord_list

    return reordered_list

if __name__=='__main__':
    a = system_from_file('/Users/zippoema/Desktop/phd/md_simulations/git_hub/input_stats/sys_ck1d_tdp43.dat')
    print(a)
