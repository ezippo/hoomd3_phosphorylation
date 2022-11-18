#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


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
    

