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
box_length=160
spacing = 32
n_tdp43s = 70

stat_file = 'input_stats/stats_module.dat'
filein_ck1d = 'input_stats/CA_ck1delta.pdb'
filein_tdp43 = 'input_stats/CA_tau.pdb'

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
    
    tdp43_id, tdp43_mass, tdp43_charge, tdp43_sigma, tdp43_pos = hu.aa_stats_sequence(filein_tdp43, aa_param_dict)
    tdp43_pos_arr = np.array(tdp43_pos)/10.
    tdp43_pos_arr = tdp43_pos_arr + 7.
    tdp43_sigma_arr = np.array(tdp43_sigma)/10.
    tdp43_length = len(tdp43_id)
    tdp43_mean_pos = ( np.sum(tdp43_pos_arr[:,0])/tdp43_length , np.sum(tdp43_pos_arr[:,1])/tdp43_length , np.sum(tdp43_pos_arr[:,2])/tdp43_length  )
    tdp43_rel_pos = tdp43_pos_arr - tdp43_mean_pos
    print(ck1d_length)
    print(tdp43_length)

    # positions
    xx = [-box_length/2 +int(spacing/2) +spacing*i for i in range(5)]
    cpos = list(itertools.product(xx, repeat=3))
    cpos = cpos[:n_tdp43s+1]
    positions = []
    for x in cpos[:-1]:
        tmp = list(tdp43_rel_pos+x)
        positions = positions + tmp
    positions = positions + [list(cpos[-1])]
    print(len(positions))
    print(len([positions[-1]]+ positions[:-1]))
    
    # ck1d moment of inertia
    I = hu.protein_moment_inertia(ck1d_rel_pos, ck1d_mass)
    I_diag, E_vec = np.linalg.eig(I)
    ck1d_diag_pos = np.dot( E_vec.T, ck1d_rel_pos.T).T
    I_check = hu.protein_moment_inertia(ck1d_diag_pos, ck1d_mass)  #check
    
    # Initialize bond
    bond_pairs=[]
    for i_tdp in range(n_tdp43s):
        bond_pairs = bond_pairs + [[i_tdp*tdp43_length+1+i, i_tdp*tdp43_length+1+i+1] for i in range(tdp43_length-1)]

    print(len(bond_pairs))
    # Now we can build HOOMD data structure for one single frame
    s=gsd.hoomd.Snapshot()
    s.particles.N = n_tdp43s*tdp43_length + 1
    s.particles.types = aa_type+['R']
    s.particles.typeid = [len(aa_type)] + tdp43_id*n_tdp43s
    s.particles.mass = [ck1d_tot_mass] + tdp43_mass*n_tdp43s
    s.particles.charge = [0] + tdp43_charge*n_tdp43s
    s.particles.position = [positions[-1]] + positions[:-1]
    s.particles.moment_inertia = [I_diag[0], I_diag[1], I_diag[2]] + [0,0,0]*tdp43_length*n_tdp43s 
    s.particles.orientation = [(1, 0, 0, 0)] * (n_tdp43s*tdp43_length+1)
    s.particles.body = [0] + [-1]*tdp43_length*n_tdp43s
    
    s.bonds.N = len(bond_pairs)
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = [0]*len(bond_pairs)
    s.bonds.group = bond_pairs
    
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0
    
    with gsd.hoomd.open(name='input_stats/ck1d-center_multi-tau_start.gsd', mode='wb') as fout:
        fout.append(s)
        

    # Defining rigid body
    import hoomd, hoomd.md     # version 2
    hoomd.context.initialize()
    system = hoomd.init.read_gsd('input_stats/ck1d-center_multi-tau_start.gsd')
    all_p = hoomd.group.all()
    
    rigid = hoomd.md.constrain.rigid()
    rigid.set_param('R', types=[aa_type[ck1d_id[i]] for i in range(ck1d_length)],
                    positions=ck1d_rel_pos)
  #  print(rigid.create_bodies(False))
    rigid.create_bodies()
     
    hoomd.dump.gsd('input_stats/ck1d-rigid_multi-tau_start.gsd', period=1, group=all_p, truncate=True)
    hoomd.run_upto(1, limit_hours=24)

    '''
    # create rigid body    
    import hoomd
    
    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_gsd(filename='input_stats/ck1d-center_multi-tau_start.gsd')
    
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
    hoomd.write.GSD.write(state=sim.state, filename='input_stats/ck1d-rigid_multi-tau_start.gsd')
    '''
    
