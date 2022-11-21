#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import hoomd
import gsd, gsd.hoomd

import hoomd_util as hu

# UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
#        mass -> amu
#        energy -> kJ/mol
# ### MACROs
production_dt=0.01 # Time step for production run in picoseconds
production_steps=200000000 # Total number of steps 
production_T=300 # Temperature for production run in Kelvin
box_lenght=50

stat_file = 'input_stats/stats_module.dat'
filein_ck1d = 'input_stats/CA_ck1delta.pdb'
ex_number = sys.argv[1]
file_start = 'input_stats/ck1d-rigid_tdp43_start.gsd'
logfile = 'ck1d-rigid_tdp43_ex'+str(ex_number)

# --------------------------- MAIN ------------------------------

if __name__=='__main__':
    # ### Input parameters for all the amino acids 
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

    ck1d_id, ck1d_mass, ck1d_charge, ck1d_sigma, ck1d_pos = hu.aa_stats_sequence(filein_ck1d, aa_param_dict)
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)       
    ck1d_tot_mass = np.sum(ck1d_mass)   
    ck1d_cof_pos = ( np.sum(ck1d_pos_arr[:,0]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,1]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,2]*ck1d_mass)/ck1d_tot_mass  )
    ck1d_rel_pos = ck1d_pos_arr - ck1d_cof_pos
    
    # ### HOOMD3 routine
    # Objects initialization
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device)
    integrator = hoomd.md.Integrator(production_dt)
    
    sim.create_state_from_gsd(filename=file_start)
    
    
    
