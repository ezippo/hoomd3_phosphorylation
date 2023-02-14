#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import itertools
import numpy as np
import gsd, gsd.hoomd
import hoomd

import hoomd_util as hu

# UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
#        mass -> amu
#        energy -> kJ/mol
# ### MACROs
box_length=50

stat_file = 'input_stats/stats_module.dat'
filein_ck1d = 'input_stats/CA_ck1delta_full.pdb'
filein_tdp43 = 'input_stats/CA_TDP-43_261truncated.pdb'

index_start_tail = 294


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
    # rigid body CK1d
    ck1d_id, ck1d_mass, ck1d_charge, ck1d_sigma, ck1d_pos = hu.aa_stats_sequence(filein_ck1d, aa_param_dict)
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)  

    tdp43_id, tdp43_mass, tdp43_charge, tdp43_sigma, tdp43_pos = hu.aa_stats_sequence(filein_tdp43, aa_param_dict)
    tdp43_pos_arr = np.array(tdp43_pos)/10.
    tdp43_pos_arr = tdp43_pos_arr + 8.
    tdp43_sigma_arr = np.array(tdp43_sigma)/10.
    tdp43_length = len(tdp43_id)
    tdp43_mean_pos = ( np.sum(tdp43_pos_arr[:,0])/tdp43_length , np.sum(tdp43_pos_arr[:,1])/tdp43_length , np.sum(tdp43_pos_arr[:,2])/tdp43_length  )
    tdp43_rel_pos = tdp43_pos_arr - tdp43_mean_pos


    # RIGID BODY
    rigid_id = np.array(ck1d_id[:index_start_tail])
    rigid_length = len(rigid_id)
    rigid_mass = ck1d_mass[:index_start_tail]
    rigid_charge = ck1d_charge[:index_start_tail]
    rigid_tot_mass = np.sum(rigid_mass)   
    # positions     
    rigid_pos_arr = ck1d_pos_arr[:index_start_tail,:]
    rigid_cof_pos = ( np.sum(rigid_pos_arr[:,0]*rigid_mass)/rigid_tot_mass , np.sum(rigid_pos_arr[:,1]*rigid_mass)/rigid_tot_mass , np.sum(rigid_pos_arr[:,2]*rigid_mass)/rigid_tot_mass  )
    rigid_rel_pos = rigid_pos_arr - rigid_cof_pos
    

    
    # ck1d moment of inertia
    I = hu.protein_moment_inertia(rigid_rel_pos, rigid_mass)
    print(I)
    I_diag, E_vec = np.linalg.eig(I)
    rigid_diag_pos = np.dot( E_vec.T, rigid_rel_pos.T).T
    I_check = hu.protein_moment_inertia(rigid_diag_pos, rigid_mass)  #check
    print(I_check) 
    
    # Initialize bond
    nbonds_ck1d = ck1d_length - rigid_length -1
    nbonds_tdp43 = tdp43_length-1
    bond_pairs=np.zeros((nbonds_ck1d+nbonds_tdp43,2),dtype=int)
    for i in range(0,nbonds_ck1d):
        bond_pairs[i,:] = np.array([i+1,i+2])
    for i in range(nbonds_ck1d, nbonds_ck1d+nbonds_tdp43):
        bond_pairs[i,:] = np.array([i+2,i+3])
    
    # Now we can build HOOMD data structure for one single frame
    rigid_ck1d_pos = np.append([rigid_cof_pos], ck1d_pos_arr[index_start_tail:,:])

    s=gsd.hoomd.open('ck1d-full-rigid_tdp43_start_nobond.gsd', 'rb')[0]

    s.bonds.N += 1
    s.bonds.typeid = np.append(s.bonds.typeid, 0)
    s.bonds.group = np.append(s.bonds.group, [[1, ck1d_length+tdp43_length]], axis=0)
    
    with gsd.hoomd.open(name='ck1d-full-rigid_tdp43_start.gsd', mode='wb') as f:
        f.append(s)
        f.close()
