#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import time
import numpy as np
import hoomd
import gsd, gsd.hoomd

import hoomd_util as hu

# UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
#        mass -> amu
#        energy -> kJ/mol

# ### MACROs
# Simulation parameters
production_dt=0.01       # Time step for production run in picoseconds
production_steps=400001   # Total number of steps 
production_T=300         # Temperature for production run in Kelvin
temp = production_T*0.00831446      # Temp is RT [kJ/mol]
box_lenght=50
seed = 3996 #np.random.randint(0, 65535) 
CONTACT = 1.33

# Files
stat_file = 'input_stats/stats_module.dat'
filein_ck1d = 'input_stats/CA_ck1delta.pdb'
#ex_number = sys.argv[1]
ex_number = 4
file_start = 'input_stats/ck1d-rigid_tdp43_start.gsd'
logfile = 'ck1d-rigid_tdp43_ex'+str(ex_number)

# Logging time interval
dt_dump = 250
dt_active_ser = 200
dt_log = 1000
dt_backup = 1000

def compute_distances(active_pos, ser_pos):
        n_active = len(active_pos)
        n_ser = len(ser_pos)
        active_center = np.array([ np.sum(active_pos[:,i])/n_active for i in range(3) ])
        distances = ser_pos-active_center
        distances = np.array([ np.sqrt( (distances[i]**2).sum() ) for i in range(n_ser)  ])
        return distances


# ### CUSTOM ACTIONS
class PrintTimestep(hoomd.custom.Action):

    def __init__(self, t_start):
        self._t_start = t_start

    def act(self, timestep):
        current_time = time.time()
        current_time = current_time - self._t_start
        print(f"Elapsed time {current_time} | Step {timestep}/{production_steps} " )

class ChangeSerine(hoomd.custom.Action):

    def __init__(self, active_serials, ser_serials):
        self._active_serials = active_serials
        self._ser_serials = ser_serials

    def act(self, timestep):
        snap = self._state.get_snapshot()
        positions = snap.particles.position
        dist = compute_distances(positions[self._active_serials], positions[self._ser_serials])
        if np.min(dist)<CONTACT:
            ser_index = np.argmin(dist)
            if ser_serials[ser_index]==15:
                snap.particles.typeid[ser_serials[ser_index]] = 20
                print(f"Phosphorylation occured: SER id {ser_serials[ser_index]}")
            elif ser_serials[ser_index]==20:
                print(f"SER {ser_serials[ser_index]} already phosphorylated")
            else:
                print(f"ERROR: residue {ser_serials[ser_index]} is not a serine! ")
        self._state.set_snapshot(snap)
        

# --------------------------- MAIN ------------------------------

if __name__=='__main__':
    # TIME START
    time_start = time.time()

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
        aa_sigma.append(aa_param_dict[k][2]/10.)
        aa_lambda.append(aa_param_dict[k][3])

    ck1d_id, ck1d_mass, ck1d_charge, ck1d_sigma, ck1d_pos = hu.aa_stats_sequence(filein_ck1d, aa_param_dict)
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)       
    ck1d_tot_mass = np.sum(ck1d_mass)   
    ck1d_cof_pos = ( np.sum(ck1d_pos_arr[:,0]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,1]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,2]*ck1d_mass)/ck1d_tot_mass  )
    ck1d_rel_pos = ck1d_pos_arr - ck1d_cof_pos
    
    # ### HOOMD3 routine
    # ## INITIALIZATION
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=seed)    
    sim.create_state_from_gsd(filename=file_start)
    snap = sim.state.get_snapshot()
    ck1d_mass = snap.particles.mass[0]
    
    type_id = snap.particles.typeid
    ser_serials = np.where(type_id[:155]==15)[0]
    activeCK1d_serials = [171, 204, 301, 302, 303, 304, 305]
    
    # # rigid body
    rigid = hoomd.md.constrain.Rigid()
    rigid.body['R'] = {
        "constituent_types": [aa_type[ck1d_id[i]] for i in range(ck1d_length)],
        "positions": ck1d_rel_pos,
        "orientations": [(1,0,0,0)]*ck1d_length,
        "charges": ck1d_charge,
        "diameters": [0.0]*ck1d_length
        }
    
    # # groups
    all_group = hoomd.filter.All()
    moving_group = hoomd.filter.Rigid(("center", "free"))
    ser_group = hoomd.filter.Intersection(moving_group, hoomd.filter.Type(["SER"]))
    active_group = hoomd.filter.Tags(activeCK1d_serials)
    active_ser_group = hoomd.filter.Union(active_group, ser_group)
    
    # ## PAIR INTERACTIONS
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond', 'body'))
    
    # # bonds
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params['AA_bond'] = dict(k=8360, r0=0.381)
    
    # # electrostatics forces
    yukawa = hoomd.md.pair.Yukawa(nlist=cell)
    for i in range(len(aa_type)):
        atom1 = aa_type[i]
        for j in range(i,len(aa_type)):
            atom2 = aa_type[j]
            yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*1.73136, kappa=1.0)
            yukawa.r_cut[(atom1,atom2)] = 3.5
        yukawa.params[(atom1,'R')] = dict(epsilon=0, kappa=1.0)
        yukawa.r_cut[(atom1,'R')] = 0.0
    yukawa.params[('R','R')] = dict(epsilon=0, kappa=1.0)
    yukawa.r_cut[('R','R')] = 0.0
    
    # # nonbonded: ashbaugh-hatch potential
    ashbaugh_table = hoomd.md.pair.Table(nlist=cell)
    for i in range(len(aa_type)):
        atom1 = aa_type[i]
        for j in range(i,len(aa_type)):
            atom2 = aa_type[j]
            Ulist = hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                      lambda_hps=[aa_lambda[i], aa_lambda[j]],
                                      r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368)
            Flist = hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                      lambda_hps=[aa_lambda[i], aa_lambda[j]],
                                      r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368)
            ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0.4, U=Ulist, F=Flist)
            ashbaugh_table.r_cut[(atom1, atom2)] = 2.0            
        ashbaugh_table.params[(atom1, 'R')] = dict(r_min=0., U=[0], F=[0])
        ashbaugh_table.r_cut[(atom1, 'R')] = 0 
    ashbaugh_table.params[('R', 'R')] = dict(r_min=0., U=[0], F=[0])
    ashbaugh_table.r_cut[('R', 'R')] = 0 
    
    # ## INTEGRATOR
    integrator = hoomd.md.Integrator(production_dt)        
    # method : Langevin
    langevin = hoomd.md.methods.Langevin(filter=moving_group, kT=temp)
    for i,name in enumerate(aa_type):
        langevin.gamma[name] = aa_mass[i]/1000.0
        langevin.gamma_r[name] = (0.0, 0.0, 0.0)
    langevin.gamma['R'] = ck1d_mass/1000.0
    langevin.gamma_r['R'] = (0.0, 0.0, 0.0)
    # constraints : rigid body
    integrator.rigid = rigid
    # forces 
    integrator.forces.append(harmonic)
    integrator.forces.append(yukawa)
    #integrator.forces.append(ashbaugh_table)
    integrator.methods.append(langevin)
    
    # ## LOGGING
    # dump files
    dump_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_dump), 
                               filename=logfile+'_dump.gsd', filter=all_group,
                               dynamic=['momentum'])                  # you can add [attributes(particles/typeid)] to trace phosphorylation
    active_ser_dcd = hoomd.write.DCD(trigger=hoomd.trigger.Periodic(dt_active_ser),
                                     filename='activeCK1d_SER_exl'+str(ex_number)+'_dump.dcd',
                                     filter=active_ser_group)
    active_ser_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_active_ser),
                                     filename='activeCK1d_SER_exl'+str(ex_number)+'_dump.gsd',
                                     filter=active_ser_group)
    # back-up files
    sim_info_log = hoomd.logging.Logger()
    sim_info_log.add(sim)
    backup1_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_backup), 
                                  filename='restart_tmp1_exl'+str(ex_number)+'.gsd', filter=all_group,
                                  mode='wb', truncate=True, log=sim_info_log)
    backup2_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_backup, phase=int(dt_backup/2.)), 
                                  filename='restart_tmp2_exl'+str(ex_number)+'.gsd', filter=all_group,
                                  mode='wb', truncate=True, log=sim_info_log)
    
    # thermodynamical quantities
    therm_quantities = hoomd.md.compute.ThermodynamicQuantities(filter=all_group)
    tq_log = hoomd.logging.Logger()
    tq_log.add(therm_quantities)
    tq_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_log), 
                             filename=logfile+'_log.gsd', filter=hoomd.filter.Null(),
                             log=tq_log)
    
    # # Custom action
    print(f"Initial time: {time.time()-time_start}")
    time_start = time.time()
    time_action = PrintTimestep(time_start)
    time_writer = hoomd.write.CustomWriter(action=time_action, trigger=hoomd.trigger.Periodic(10000))
    changeser_action = ChangeSerine(active_serials=activeCK1d_serials, ser_serials=ser_serials)
    changeser_updater = hoomd.update.CustomUpdater(action=changeser_action, trigger=hoomd.trigger.Periodic(200))

    # ## SET SIMULATION OPERATIONS
    sim.operations.integrator = integrator 
    sim.operations.computes.append(therm_quantities)
    sim.operations.writers.append(dump_gsd)
    sim.operations.writers.append(active_ser_dcd)
    sim.operations.writers.append(backup1_gsd)
    sim.operations.writers.append(backup2_gsd)
    sim.operations.writers.append(tq_gsd)
    sim.operations += time_writer
    sim.operations += changeser_updater

    sim.run(production_steps)
    