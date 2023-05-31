#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import time
import numpy as np
import hoomd
import gsd, gsd.hoomd
import logging

import hoomd_util as hu

index_start_tail = 294
contacts = []

def metropolis_boltzmann(dU, dmu, beta=2.494338):
    x = np.random.rand()
    if np.log(x) <= -beta*(dU+dmu):
        return True
    else:
        return False


# ### CUSTOM ACTIONS
class PrintTimestep(hoomd.custom.Action):

    def __init__(self, t_start):
        self._t_start = t_start

    def act(self, timestep):
        current_time = time.time()
        current_time = current_time - self._t_start
        print(f"Elapsed time {current_time} | Step {timestep}/{production_steps} " )


class ChangeSerine(hoomd.custom.Action):

    def __init__(self, active_serials, ser_serials, forces, glb_contacts, temp, Dmu):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._forces = forces
        self._glb_contacts = glb_contacts
        self._temp = temp
        self._Dmu = Dmu

    def act(self, timestep):
        snap = self._state.get_snapshot()
        positions = snap.particles.position
        active_pos = hu.compute_center(positions[self._active_serials])
        dist = hu.compute_distances(active_pos, positions[self._ser_serials])
        min_dist = np.min(dist)

        if min_dist<contact_dist:
            ser_index = self._ser_serials[np.argmin(dist)]

            if snap.particles.typeid[ser_index]==15:
                U_in = self._forces[0].energy + self._forces[1].energy
                snap.particles.typeid[ser_index] = 20
                self._state.set_snapshot(snap)
                U_fin = self._forces[0].energy + self._forces[1].energy
                logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                if metropolis_boltzmann(U_fin-U_in, self._Dmu, self._temp):
                    logging.info(f"Phosphorylation occured: SER id {ser_index}")
                    self._glb_contacts += [[timestep, ser_index, 1, min_dist, U_fin-U_in]]
                else:
                    snap.particles.typeid[ser_index] = 15
                    self._state.set_snapshot(snap)
                    logging.info(f'Phosphorylation SER id {ser_index} not accepted')
                    self._glb_contacts += [[timestep, ser_index, 0, min_dist, U_fin-U_in]]
                    
            elif snap.particles.typeid[ser_index]==20:
                U_in = self._forces[0].energy + self._forces[1].energy
                snap.particles.typeid[ser_index] = 15
                self._state.set_snapshot(snap)
                U_fin = self._forces[0].energy + self._forces[1].energy
                logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                if metropolis_boltzmann(U_fin-U_in, -self._Dmu, self._temp):
                    logging.info(f"Dephosphorylation occured: SER id {ser_index}")
                    self._glb_contacts += [[timestep, ser_index, -1, min_dist, U_fin-U_in]]
                else:
                    snap.particles.typeid[ser_index] = 20
                    self._state.set_snapshot(snap)
                    logging.info(f'Dephosphorylation SER id {ser_index} not accepted')
                    self._glb_contacts += [[timestep, ser_index, 2, min_dist, U_fin-U_in]]

            else:
                raise Exception(f"Residue {ser_index} is not a serine!")


class ContactsBackUp(hoomd.custom.Action):

    def __init__(self, glb_contacts):
        self._glb_contacts = glb_contacts

    def act(self, timestep):
        np.savetxt(logfile+"_contactsBCKP.txt", self._glb_contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")

                

# --------------------------- MAIN ------------------------------

if __name__=='__main__':
    # TIME START
    time_start = time.time()
    logging.basicConfig(level=logging.WARNING)

    # UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
    #        mass -> amu
    #        energy -> kJ/mol
    #
    # ### MACROs from file
    input_file = sys.argv[1]
    macro_dict = hu.macros_from_file(input_file)
    # Simulation parameters
    production_dt = float(macro_dict['production_dt'])        # Time step for production run in picoseconds
    production_steps = int(macro_dict['production_steps'])                       # Total number of steps 
    production_T = float(macro_dict['production_T'])                      # Temperature for production run in Kelvin
    temp = production_T * 0.00831446                  # Temp is RT [kJ/mol]
    box_lenght = int(macro_dict['box_lenght'])
    start = int(macro_dict['start'])	                           # 0 -> new simulation, 1 -> restart
    contact_dist = float(macro_dict['contact_dist'])
    Dmu = float(macro_dict['Dmu'])
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
    dt_try_change = int(macro_dict['dt_try_change'])
    dt_time = int(macro_dict['dt_time'])
    dt_active_ser = int(macro_dict['dt_active_ser'])

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
    ck1d_pos = np.array(ck1d_pos)/10.

    # RIGID BODY
    rigid_id = np.array(ck1d_id[:index_start_tail])
    rigid_length = len(rigid_id)
    rigid_mass = ck1d_mass[:index_start_tail]
    rigid_charge = ck1d_charge[:index_start_tail]
    rigid_tot_mass = np.sum(rigid_mass)   
    # positions     
    rigid_pos_arr = ck1d_pos[:index_start_tail,:]
    rigid_cof_pos = ( np.sum(rigid_pos_arr[:,0]*rigid_mass)/rigid_tot_mass , np.sum(rigid_pos_arr[:,1]*rigid_mass)/rigid_tot_mass , np.sum(rigid_pos_arr[:,2]*rigid_mass)/rigid_tot_mass  )
    rigid_rel_pos = rigid_pos_arr - rigid_cof_pos
    
    
    # ### HOOMD3 routine
    # ## INITIALIZATION
    device = hoomd.device.GPU(notice_level=2)
    sim = hoomd.Simulation(device=device, seed=seed)
    if start==0:
        traj = gsd.hoomd.open(file_start)
        snap = traj[0]
        snap.configuration.step = 0
        sim.create_state_from_snapshot(snapshot=snap)
    elif start==1:
        sim.create_state_from_gsd(filename=file_start)
        snap = sim.state.get_snapshot()
    init_step = sim.initial_timestep

    ck1d_mass = snap.particles.mass[0]
    
    rbody = np.where(snap.particles.body==0)[0]
    print(rbody)
    #print([aa_type[snap.particles.typeid[i]] for i in rbody])
    #print([aa_type[rigid_id[i]] for i in range(rigid_length)])
    
    type_id = snap.particles.typeid

    ser_serials = np.where(type_id[122:276]==15)[0]
    ser_serials = ser_serials+122
    activeCK1d_serials = [424, 425, 426]
    
    # # rigid body    
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["R"] = {
        "constituent_types": [aa_type[rigid_id[i]] for i in range(rigid_length)],
        "positions": rigid_rel_pos,
        "orientations": [(1,0,0,0)]*rigid_length,
        "charges": rigid_charge,
        "diameters": [0.0]*rigid_length
        }
        
    # # groups
    all_group = hoomd.filter.All()
    moving_group = hoomd.filter.Rigid(("center", "free"))
    ser_group = hoomd.filter.Tags(list(ser_serials))
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
            ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0.2, U=Ulist, F=Flist)
            ashbaugh_table.r_cut[(atom1, atom2)] = 2.0            
        ashbaugh_table.params[(atom1, 'R')] = dict(r_min=0., U=[0], F=[0])
        ashbaugh_table.r_cut[(atom1, 'R')] = 0 
    ashbaugh_table.params[('R', 'R')] = dict(r_min=0., U=[0], F=[0])
    ashbaugh_table.r_cut[('R', 'R')] = 0 
    
    # ## INTEGRATOR
    integrator = hoomd.md.Integrator(production_dt, integrate_rotational_dof=True)        
    # method : Langevin
    langevin = hoomd.md.methods.Langevin(filter=moving_group, kT=temp)
    for i,name in enumerate(aa_type):
        langevin.gamma[name] = aa_mass[i]/1000.0
        langevin.gamma_r[name] = (0.0, 0.0, 0.0)
    langevin.gamma['R'] = ck1d_mass/1000.0
    langevin.gamma_r['R'] = (1.0, 1.0, 1.0)
    # constraints : rigid body
    integrator.rigid = rigid
    # forces 
    integrator.forces.append(harmonic)
    integrator.forces.append(yukawa)
    integrator.forces.append(ashbaugh_table)
    integrator.methods.append(langevin)
    
    # ## LOGGING
    # dump files
    dump_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_dump), 
                               filename=logfile+'_dump.gsd', filter=all_group,
                               dynamic=['property', 'momentum', 'attribute', 'topology'])                  # you can add [attributes(particles/typeid)] to trace phosphorylation
    #active_ser_dcd = hoomd.write.DCD(trigger=hoomd.trigger.Periodic(dt_active_ser),
    #                                 filename='activeCK1d_SER_exl'+str(ex_number)+'_dump.dcd',
    #                                 filter=active_ser_group)
    active_ser_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_active_ser),
                                     filename=logfile+'_activeCK1d_SER_dump.gsd',
                                     filter=active_ser_group)
    #start_active_ser_gsd = hoomd.write.GSD(trigger=hoomd.trigger.On(1),
    #                                 filename='activeCK1d_SER_start.gsd',
    #                                 filter=active_ser_group)                       # save starting config of activeCK1d and SERs
    
    # back-up files
    sim_info_log = hoomd.logging.Logger()
    sim_info_log.add(sim)
    backup1_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_backup), 
                                  filename=logfile+'_restart1.gsd', filter=all_group,
                                  mode='wb', truncate=True, log=sim_info_log)
    backup2_gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(dt_backup, phase=int(dt_backup/2.)), 
                                  filename=logfile+'_restart2.gsd', filter=all_group,
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
    time_writer = hoomd.write.CustomWriter(action=time_action, trigger=hoomd.trigger.Periodic(dt_time))
    
    changeser_action = ChangeSerine(active_serials=activeCK1d_serials, ser_serials=ser_serials, forces=[yukawa, ashbaugh_table], glb_contacts=contacts, temp=temp, Dmu=Dmu)
    changeser_updater = hoomd.update.CustomUpdater(action=changeser_action, trigger=hoomd.trigger.Periodic(dt_try_change))

    contacts_action = ContactsBackUp(glb_contacts=contacts)
    contacts_bckp_writer = hoomd.write.CustomWriter(action=contacts_action, trigger=hoomd.trigger.Periodic(dt_backup))
    
    # ## SET SIMULATION OPERATIONS
    sim.operations.integrator = integrator 
    sim.operations.computes.append(therm_quantities)
    
    sim.operations.writers.append(dump_gsd)
    sim.operations.writers.append(active_ser_gsd)
    #sim.operations.writers.append(start_active_ser_gsd)     # save starting config of activeCK1d and SERs
    sim.operations.writers.append(backup1_gsd)
    sim.operations.writers.append(backup2_gsd)
    sim.operations.writers.append(tq_gsd)
    sim.operations += time_writer
    sim.operations += changeser_updater
    sim.operations += contacts_bckp_writer

    sim.run(production_steps-init_step)
    
    if start==1:
        cont_prev = np.loadtxt(logfile+"_contacts.txt")
        if len(cont_prev)!=0:
                contacts = np.append(cont_prev, contacts, axis=0)
    np.savetxt(logfile+"_contacts.txt", contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")
    
    hoomd.write.GSD.write(state=sim.state, filename=logfile+'_end.gsd')
    
    
