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
box_length=200
spacing = 25
n_tdp43s = 200

stat_file = 'input_stats/stats_module.dat'
filein_ck1d = 'input_stats/CA_ck1delta.pdb'

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
    ck1d_id, ck1d_mass, ck1d_charge, ck1d_sigma, ck1d_pos = hu.aa_stats_sequence(filein_ck1d, aa_param_dict)
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)       
    ck1d_tot_mass = np.sum(ck1d_mass)   
    ck1d_cof_pos = ( np.sum(ck1d_pos_arr[:,0]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,1]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,2]*ck1d_mass)/ck1d_tot_mass  )
    ck1d_rel_pos = ck1d_pos_arr - ck1d_cof_pos

        
    # create rigid body    
    
    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_gsd(filename='input_stats/ck1d-center_multi-tdp43_start.gsd')
    
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["R"] = {
        "constituent_types": [aa_type[ck1d_id[i]] for i in range(ck1d_length)],
        "positions": ck1d_rel_pos,
        "orientations": [(1,0,0,0)]*ck1d_length,
        "charges": ck1d_charge,
        "diameters": [0.0]*ck1d_length
        }

    rigid.create_bodies(sim.state)
    
    integrator = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=True)
    integrator.rigid = rigid
    sim.operations.integrator = integrator
    
    sim.run(0)
    hoomd.write.GSD.write(state=sim.state, filename='input_stats/ck1d-rigid_multi-tdp43_start.gsd')
    
