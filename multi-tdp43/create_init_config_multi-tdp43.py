#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import itertools
import numpy as np
import gsd, gsd.hoomd

import hoomd_util as hu

# UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
#        mass -> amu
#        energy -> kJ/mol
# ### MACROs
box_length=200
spacing = 25
n_tdp43s = 200

stat_file = 'input_stats/stats_module.dat'
filein_tdp43 = 'input_stats/CA_TDP-43_261truncated.pdb'

# ------------------------- MAIN -------------------------------

if __name__=='__main__':
    # Input parameters for all the amino acids 
    aa_param_dict = hu.aa_stats_from_file(stat_file)
    aa_type = list(aa_param_dict.keys())    
    aa_mass = []
    aa_charge = []
    aa_sigma = []
    aa_lambda = []
    for k in aa_type:
        aa_mass.append(aa_param_dict[k][0])
        aa_charge.append(aa_param_dict[k][1])
        aa_sigma.append(aa_param_dict[k][2])
        aa_lambda.append(aa_param_dict[k][3])
    
    # Now we can translate the entire sequence into a sequence of numbers and 
    # assign to each a.a. of the sequence its stats  
    tdp43_id, tdp43_mass, tdp43_charge, tdp43_sigma, tdp43_pos = hu.aa_stats_sequence(filein_tdp43, aa_param_dict)
    tdp43_pos_arr = np.array(tdp43_pos)/10.
    tdp43_pos_arr = tdp43_pos_arr + 7.
    tdp43_sigma_arr = np.array(tdp43_sigma)/10.
    tdp43_length = len(tdp43_id)
    tdp43_mean_pos = ( np.sum(tdp43_pos_arr[:,0])/tdp43_length , np.sum(tdp43_pos_arr[:,1])/tdp43_length , np.sum(tdp43_pos_arr[:,2])/tdp43_length  )
    tdp43_rel_pos = tdp43_pos_arr - tdp43_mean_pos
    print(tdp43_length)

    # positions
    xx = [-box_length/2 +20 +25*i for i in range(7)]
    cpos = list(itertools.product(xx, repeat=3))
    cpos = cpos[:n_tdp43s]
    positions = []
    for x in cpos:
        tmp = list(tdp43_rel_pos+x)
        positions = positions + tmp
    print(len(positions))
    
    # Initialize bond
    bond_pairs=[]
    for i_tdp in range(n_tdp43s):
        bond_pairs = bond_pairs + [[i_tdp*tdp43_length+i, i_tdp*tdp43_length+i+1] for i in range(tdp43_length-1)]
    print(len(bond_pairs))

    # Now we can build HOOMD data structure for one single frame
    s=gsd.hoomd.Snapshot()
    s.particles.N = n_tdp43s*tdp43_length
    s.particles.types = aa_type
    s.particles.typeid = tdp43_id*n_tdp43s
    s.particles.mass = tdp43_mass*n_tdp43s
    s.particles.charge = tdp43_charge*n_tdp43s
    s.particles.position = positions
    
    s.bonds.N = len(bond_pairs)
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = [0]*len(bond_pairs)
    s.bonds.group = bond_pairs
    
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0
    
    with gsd.hoomd.open(name='input_stats/multi-tdp43_start.gsd', mode='wb') as fout:
        fout.append(s)
 