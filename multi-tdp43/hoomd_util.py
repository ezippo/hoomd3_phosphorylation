#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def macros_from_file(infile):
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
            if line[0]!='#' and line[0]!='\n':
                line_list = line.rsplit()
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


def compute_distances(active_center, ser_pos):
    n_ser = len(ser_pos)
    distances = ser_pos-active_center
    distances = np.array([ np.sqrt( (distances[i]**2).sum() ) for i in range(n_ser)  ])
    return distances

def compute_center(pos):
    n = len(pos)
    center_pos = np.array([ np.sum(pos[:,i])/n for i in range(3) ])
    return center_pos


if __name__=='__main__':
    dic = macros_from_file('input_test1.dat')
    tstep = dic['production_dt']
    b_n = int(dic['box_lenght'])
    print(tstep)
    print(b_n)