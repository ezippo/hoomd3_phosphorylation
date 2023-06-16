#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import hoomd
from hoomd import azplugins
import gsd

from hoomd_util import *


# --------------------------- MAIN ------------------------------

if __name__=='__main__':
    # UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
    #        mass -> amu
    #        energy -> kJ/mol
    #
    # ### MACROs from file
    input_file = sys.argv[1]
    macro_dict = macros_from_file(input_file)
    # Simulation parameters
    production_dt = float(macro_dict['production_dt'])        # Time step for production run in picoseconds
    production_steps = int(macro_dict['production_steps'])                       # Total number of steps 
    production_T = float(macro_dict['production_T'])                      # Temperature for production run in Kelvin
    temp = production_T * 0.00831446                  # Temp is RT [kJ/mol]
    box_lenght = int(macro_dict['box_lenght'])
    start = int(macro_dict['start'])	                           # 0 -> new simulation, 1 -> restart
    seed = int(macro_dict['seed'])
    # Files
    stat_file = macro_dict['stat_file']
    filein_ck1d = macro_dict['filein_ck1d']
    file_start = macro_dict['file_start']
    logfile = macro_dict['logfile']
    # Logging time interval
    dt_dump = int(macro_dict['dt_dump'])
    dt_log = int(macro_dict['dt_log'])
    dt_backup = int(macro_dict['dt_backup'])

    # ### Input parameters for all the amino acids 
    aa_param_dict = aa_stats_from_file(stat_file)
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

    ck1d_id, ck1d_mass, ck1d_charge, ck1d_sigma, ck1d_pos = aa_stats_sequence(filein_ck1d, aa_param_dict)
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)       
    ck1d_tot_mass = np.sum(ck1d_mass)   
    ck1d_cof_pos = ( np.sum(ck1d_pos_arr[:,0]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,1]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,2]*ck1d_mass)/ck1d_tot_mass  )
    ck1d_rel_pos = ck1d_pos_arr - ck1d_cof_pos
    
    # ### HOOMD2 old version routine
    # ## INITIALIZATION
    hoomd.context.initialize()
    system = hoomd.init.read_gsd(file_start)
    snap = system.take_snapshot()
    ck1d_mass = snap.particles.mass[0]
    
    # # rigid body
    rigid = hoomd.md.constrain.rigid()
    rigid.set_param('R', types=[aa_type[ck1d_id[i]] for i in range(ck1d_length)],
                    positions=ck1d_rel_pos)

    # # groups
    all_group = hoomd.group.all()
    center_group = hoomd.group.rigid_center()
    nonrigid_group = hoomd.group.nonrigid()
    moving_group = hoomd.group.union('moving_group', center_group, nonrigid_group)
    
    ser_group = hoomd.group.intersection('SER_group', nonrigid_group, hoomd.group.type('SER'))
    active_group = hoomd.group.tag_list('active_group', tags=[301, 302, 303])
    active_ser_group = hoomd.group.union('activeCK1d_SER_group', active_group, ser_group)
    
    # ## PAIR INTERACTIONS
    cell = hoomd.md.nlist.cell()

    ## Bonds
    harmonic = hoomd.md.bond.harmonic()
    harmonic.bond_coeff.set('AA_bond', k=8360, r0=0.381)
    
    ## Nonbonded
    cell.reset_exclusions(exclusions=['1-2', 'body'])
    nb_pair = azplugins.pair.ashbaugh(r_cut=0, nlist=cell)
    for i in aa_type:
        for j in aa_type:
            nb_pair.pair_coeff.set(i,j,lam=(aa_param_dict[i][3]+aa_param_dict[j][3])/2.,
                              epsilon=0.8368,
                              sigma=(aa_param_dict[i][2]+aa_param_dict[j][2])/10./2.,
                              r_cut=2.0) 
        nb_pair.pair_coeff.set(i,'R',lam=0,epsilon=0,sigma=0,r_cut=0)
        nb_pair.pair_coeff.set('R',i,lam=0,epsilon=0,sigma=0,r_cut=0)
    nb_pair.pair_coeff.set('R','R',lam=0,epsilon=0,sigma=0,r_cut=0)
    
    ## Electrostatics
    yukawa = hoomd.md.pair.yukawa(r_cut=0.0, nlist=cell)
    for i,atom1 in enumerate(aa_type):
        for j,atom2 in enumerate(aa_type):
            yukawa.pair_coeff.set(atom1,atom2,epsilon=aa_param_dict[atom1][1]*aa_param_dict[atom2][1]*1.73136, kappa=1.0, r_cut=3.5)
        yukawa.pair_coeff.set(atom1,'R',epsilon=0, kappa=1.0, r_cut=0)
        yukawa.pair_coeff.set('R',atom1,epsilon=0, kappa=1.0, r_cut=0)
    yukawa.pair_coeff.set('R','R',epsilon=0, kappa=1.0, r_cut=0)
    
    # ## INTEGRATOR: Langevin
    hoomd.md.integrate.mode_standard(dt=production_dt) # Time units in ps
    temp = production_T*0.00831446
    integrator = hoomd.md.integrate.langevin(group=moving_group, kT=temp, seed=3996) # Temp is kT/0.00831446
    for i,name in enumerate(aa_type):
        integrator.set_gamma(name,gamma=aa_mass[i]/1000.0)
    integrator.set_gamma('R', gamma=ck1d_mass/1000.0)
    integrator.set_gamma_r('R', gamma_r=(4.0,4.0,4.0))
 
    # ## LOGGING
    # dump files
    hoomd.dump.gsd(logfile+'_dump.gsd', period=dt_dump, group=all_group, overwrite=False)  # trajectory

    # back-up files
    hoomd.dump.gsd(logfile+'_restart1.gsd', period=dt_backup, group=all_group, truncate=True)                         # backup1
    hoomd.dump.gsd(logfile+'_restart2.gsd', period=dt_backup, group=all_group, truncate=True, phase=int(dt_backup/2))   # backup2
    
    # thermodynamical quantities
    hoomd.analyze.log(filename=logfile+'.log', quantities=['potential_energy', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'temperature','lx','ly','lz'], 
                      period=dt_log, overwrite=False, header_prefix='#')
    hoomd.analyze.log(filename=logfile+'_stress.log', quantities=['pressure_xy', 'pressure_xz', 'pressure_yz'], 
                      period=dt_log, overwrite=False, header_prefix='#')

    #hoomd.dump.dcd('activeCK1d_SER_sim'+str(ex_number)+'_dump.dcd', period=200, group=active_ser_group, overwrite=False)         # SER and ASP149 trajectory
    
    ## Run simulation
    hoomd.run(production_steps, limit_hours=24)

    hoomd.dump.gsd(logfile+'_end.gsd', period=1, group=all_group, truncate=True, dynamic=['momentum'])  
    hoomd.run(1)
    