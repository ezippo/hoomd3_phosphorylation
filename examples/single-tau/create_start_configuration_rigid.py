# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import gsd, gsd.hoomd
import hoomd, hoomd.md

from hoomd_util import *

# UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
#        mass -> amu
#        energy -> kJ/mol
# ### MACROs
production_dt=0.01 # Time step for production run in picoseconds
box_length=50

stat_file = 'input_stats/stats_module.dat'
filein_ck1d = 'input_stats/CA_ck1delta.pdb'
filein_tau = 'input_stats/CA_tau.pdb'

if __name__=='__main__':
    # Input parameters for all the amino acids 
    aa_param_dict = aa_stats_from_file(stat_file)
    aa_type = list(aa_param_dict.keys())    
    aa_mass = []
    aa_charge = []
    aa_sigma = []
    aa_lambda = []
    for k in aa_type:
        aa_mass.append(aa_param_dict[k][0])
        aa_charge.append(aa_param_dict[k][1])
        aa_sigma.append(aa_param_dict[k][2]/10.)
        aa_lambda.append(aa_param_dict[k][3])
    
    # Now we can translate the entire sequence into a sequence of numbers and 
    # assign to each a.a. of the sequence its stats
    ck1d_id, ck1d_mass, ck1d_charge, ck1d_sigma, ck1d_pos = aa_stats_sequence(filein_ck1d, aa_param_dict)
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)       
    ck1d_tot_mass = np.sum(ck1d_mass)   
    ck1d_cof_pos = ( np.sum(ck1d_pos_arr[:,0]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,1]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,2]*ck1d_mass)/ck1d_tot_mass  )
    ck1d_rel_pos = ck1d_pos_arr - ck1d_cof_pos
    
    tdp43_id, tdp43_mass, tdp43_charge, tdp43_sigma, tdp43_pos = aa_stats_sequence(filein_tau, aa_param_dict)
    tdp43_pos_arr = np.array(tdp43_pos)/10.
    tdp43_pos_arr = tdp43_pos_arr + 10.
    tdp43_sigma_arr = np.array(tdp43_sigma)/10.
    tdp43_length = len(tdp43_id)
    print(ck1d_length)
    print(tdp43_length)
    
    
    # ck1d moment of inertia
    I = protein_moment_inertia(ck1d_rel_pos, ck1d_mass)
    print(I)
    I_diag, E_vec = np.linalg.eig(I)
    ck1d_diag_pos = np.dot( E_vec.T, ck1d_rel_pos.T).T
    I_check = protein_moment_inertia(ck1d_diag_pos, ck1d_mass)  #check
    print(I_check) 
    
    # Initialize bond
    nbonds_tdp43 = tdp43_length-1
    bond_pairs=np.zeros((nbonds_tdp43,2),dtype=int)
    for i in range(0,nbonds_tdp43):
        bond_pairs[i,:] = np.array([i+1,i+2])
    
    # Now we can build HOOMD data structure for one single frame
    s=gsd.hoomd.Snapshot()
    s.particles.N = tdp43_length + 1
    s.particles.types = aa_type+['R']
    s.particles.typeid = [len(aa_type)] + tdp43_id
    s.particles.mass = [ck1d_tot_mass] + tdp43_mass
    s.particles.charge = [0] + tdp43_charge
    s.particles.position = np.append( [ck1d_cof_pos] , tdp43_pos_arr)
    s.particles.moment_inertia = [I_diag[0], I_diag[1], I_diag[2]] + [0,0,0]*tdp43_length 
    s.particles.orientation = [(1, 0, 0, 0)] * (tdp43_length+1)
    s.particles.body = [0] + [-1]*tdp43_length
    
    s.bonds.N = nbonds_tdp43
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = [0]*(nbonds_tdp43)
    s.bonds.group = bond_pairs
    
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0

    with gsd.hoomd.open(name='ck1d-center_tau_start.gsd', mode='wb') as f:
        f.append(s)
        f.close()
        
    
    # Defining rigid body
    hoomd.context.initialize()
    system = hoomd.init.read_gsd('ck1d-center_tau_start.gsd')
    all_p = hoomd.group.all()
    
    snapshot = system.take_snapshot()
    print(snapshot.particles.body)
    
    rigid = hoomd.md.constrain.rigid()
    rigid.set_param('R', types=[aa_type[ck1d_id[i]] for i in range(ck1d_length)],
                    positions=ck1d_rel_pos)
  #  print(rigid.create_bodies(False))
    rigid.create_bodies()
    
    snapshot = system.take_snapshot()
    print(snapshot.particles.body)
    
    hoomd.dump.gsd('ck1d-rigid_tau_start.gsd', period=1, group=all_p, truncate=True)
    hoomd.run_upto(1, limit_hours=24)