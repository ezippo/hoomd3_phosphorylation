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

    s=gsd.hoomd.Snapshot()
    s.particles.N = nbonds_ck1d+1 + 1 + tdp43_length
    s.particles.types = aa_type+['R']
    s.particles.typeid = [len(aa_type)] + ck1d_id[index_start_tail:] + tdp43_id
    s.particles.mass = [rigid_tot_mass] + ck1d_mass[index_start_tail:] + tdp43_mass
    s.particles.charge = [0] + ck1d_charge[index_start_tail:] + tdp43_charge
    s.particles.position = np.append( rigid_ck1d_pos , tdp43_pos_arr)
    s.particles.moment_inertia = [I_diag[0], I_diag[1], I_diag[2]] + [0,0,0]*(nbonds_ck1d+1 + tdp43_length) 
    s.particles.orientation = [(1, 0, 0, 0)] * (1 + nbonds_ck1d+1 + tdp43_length)
    s.particles.body = [0] + [-1]*(nbonds_ck1d+1 + tdp43_length)
    
    s.bonds.N = nbonds_ck1d + nbonds_tdp43
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = [0]*(nbonds_ck1d + nbonds_tdp43)
    s.bonds.group = bond_pairs
    
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0

    with gsd.hoomd.open(name='ck1d-full-center_tdp43_start_try.gsd', mode='wb') as f:
        f.append(s)
        f.close()


    # create rigid body    
    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_gsd(filename='ck1d-full-center_tdp43_start_try.gsd')
    
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["R"] = {
        "constituent_types": [aa_type[rigid_id[i]] for i in range(rigid_length)],
        "positions": rigid_rel_pos,
        "orientations": [(1,0,0,0)]*rigid_length,
        "charges": rigid_charge,
        "diameters": [0.0]*rigid_length
        }

    rigid.create_bodies(sim.state)
    
    integrator = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=True)
    integrator.rigid = rigid
    sim.operations.integrator = integrator
    
    sim.run(0)
    hoomd.write.GSD.write(state=sim.state, filename='ck1d-full-rigid_tdp43_start_nobond_try.gsd')

    