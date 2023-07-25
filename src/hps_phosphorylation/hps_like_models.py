#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import time
import numpy as np
#import hoomd
import gsd, gsd.hoomd
import logging

import hps_phosphorylation.hoomd_util as hu


def create_init_configuration(syslist, aa_param_dict, box_length):
    n_mols = len(syslist)
    n_chains = np.sum([int(syslist[i]['N']) for i in range(n_mols)])
    aa_type = list(aa_param_dict.keys())
    
    s=gsd.hoomd.Snapshot()
    
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0
    s.particles.N = 0
    s.particles.types = aa_type
    s.particles.typeid = []
    s.particles.mass = []
    s.particles.charge = []
    s.particles.position = []
    s.particles.moment_inertia = [] 
    s.particles.orientation = []
    s.particles.body = []
    s.bonds.N = 0
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = []
    s.bonds.group = []
    
    K = math.ceil(n_chains**(1/3))
    spacing = box_length/K
    x = np.linspace(-box_length/2, box_length/2, K, endpoint=False)
    positions = list(itertools.product(x, repeat=3))
    positions = np.array(positions) + [spacing/2, spacing/2, spacing/2]
    np.random.shuffle(positions)
    positions = positions[:n_chains, :]
    
    n_prev_mol = 0
    n_prev_res = 0
    for mol in range(n_mols):
        mol_dict = syslist[mol]
#        add_molecule(syslist[mol], aa_param_dict)
        chain_id, chain_mass, chain_charge, chain_sigma, chain_pos = aa_stats_sequence(mol_dict['pdb'], aa_param_dict)
        chain_length = len(chain_id)
        chain_rel_pos = chain_positions_from_pdb(mol_dict['pdb'], relto='com', chain_mass=chain_mass)   # positions relative to c.o.m. 

        if mol_dict['rigid']=='0':
            mol_pos = []
            bond_pairs=[]
            n_mol_chains = int(mol_dict['N'])
            for i_chain in range(n_mol_chains):
                mol_pos += list(chain_rel_pos+positions[n_prev_mol+i_chain])
                bond_pairs += [[n_prev_res + i+i_chain*chain_length, n_prev_res + i+1+i_chain*chain_length] for i in range(chain_length-1)]

            s.particles.N += n_mol_chains*chain_length
            s.particles.typeid += n_mol_chains*chain_id
            s.particles.mass += n_mol_chains*chain_mass
            s.particles.charge += n_mol_chains*tdp43_charge
            s.particles.position +=  mol_pos
            s.particles.moment_inertia += [0,0,0]*chain_length*n_mol_chains
            s.particles.orientation += [(1, 0, 0, 0)]*chain_length*n_mol_chains
            s.particles.body += [-1]*chain_length*n_mol_chains
            
            s.bonds.N += len(bond_pairs)
            s.bonds.typeid += [0]*len(bond_pairs)
            s.bonds.group += bond_pairs
            
            n_prev_mol += n_mol_chains
            n_prev_res += n_mol_chains*chain_length
        
        else:
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
            n_rigids = len(rigid_ind_l)
            
            rigid = hoomd.md.constrain.Rigid()
            types_rigid_bodies = []
            typeid_rigid_bodies = []
            mass_rigid_bodies = []
            moment_inertia_rigid_bodies = []
            position_rigid_bodies = []
            typeid_free_bodies = []
            mass_free_bodies = []
            charge_free_bodies = []
            position_free_bodies = []
            length_free_bodies = []
            bonds_free_rigid = []
            n_prev_res=0
            for r in range(n_rigids):
                # rigid body and previous free body indexes
                rigid_ind = rigid_ind_l[r]
                free_ind = [i for i in range(n_prev_res, rigid_ind[0])]
                # rigid body properties
                types_rigid_bodies += ['R'+str(r+1)]                        # type R1, R2, R3 ...
                typeid_rigid_bodies += [len(aa_type)+len(types_rigid_bodies)-1]  
                rigid_mass = [chain_mass[i] for i in rigid_ind]             
                mass_rigid_bodies += [np.sum(rigid_mass)]                   # total mass of the rigid body
                rigid_rel_pos = chain_rel_pos[rigid_ind] 
                reshaped_rigid_mass = np.reshape( rigid_mass, (len(rigid_mass),1) )
                rigid_com_rel_pos = np.sum(rigid_rel_pos * reshaped_rigid_mass, axis=0) / np.sum(rigid_mass)       # c.o.m. relative to the center of the molecule
                rigid_rel_pos = rigid_rel_pos-rigid_com_rel_pos             # positions of monomers of the rigid body relative to the c.o.m.
                position_rigid_bodies += [rigid_com_rel_pos+positions[0]]   # position of c.o.m.
                I = protein_moment_inertia(rigid_rel_pos, rigid_mass)
                I_diag, E_vec = np.linalg.eig(I)
                moment_inertia_rigid_bodies += [I_diag[0], I_diag[1], I_diag[2]]
                # create rigid body object 
                rigid.body['R'+str(r+1)] = {
                    "constituent_types": [aa_type[chain_id[i]] for i in rigid_ind],
                    "positions": rigid_rel_pos,
                    "orientations": [(1,0,0,0)]*len(rigid_ind),
                    "charges": [ chain_charge[i] for i in rigid_ind ],
                    "diameters": [0.0]*len(rigid_ind)
                    }
                # free body properties
                if len(free_ind)==0:
                    length_free_bodies += [0]
                else:
                    typeid_free_bodies += [chain_id[i] for i in free_ind]
                    mass_free_bodies += [chain_mass[i] for i in free_ind]
                    charge_free_bodies += [chain_charge[i] for i in free_ind]
                    free_rel_pos = chain_rel_pos[free_ind]                      # positions of the free monomers relative to the center of the molecule
                    position_free_bodies += list(free_rel_pos+positions[0])     # positions of the free monomers
                    length_free_bodies += [len(free_ind)]
                    bonds_free_rigid += [ [n_rigids+np.sum(length_free_bodies)-1, r+1] ]
                    if free_ind[0]!=0:
                        bonds_free_rigid += [ [n_rigids+np.sum(length_free_bodies[:-1]), -r] ]
                    
                n_prev_res += len(free_ind) + len(rigid_ind)
        
            # free monomers in the final tail
            if rigid_ind[-1]<len(chain_id):
                free_ind = [i for i in range(n_prev_res, len(chain_id))]
                typeid_free_bodies += [chain_id[i] for i in free_ind]
                mass_free_bodies += [chain_mass[i] for i in free_ind]
                charge_free_bodies += [chain_charge[i] for i in free_ind]
                free_rel_pos = chain_rel_pos[free_ind]                      # positions of the free monomers relative to the center of the molecule
                position_free_bodies += list(free_rel_pos+positions[0])     # positions of the free monomers
                length_free_bodies += [len(free_ind)]
                bonds_free_rigid += [ [n_rigids+np.sum(length_free_bodies[:-1]), -r-1] ]






            rigid_id_l = read_rigid_indexes(mol_dict['rigid'])
            rigid_list = []
            for i in range(len(rigid_id_l)):
                rigid = hoomd.md.constrain.Rigid()
                rigid.body['R'+str(i)] = {
                    "constituent_types": [aa_type[chain_id[i]] for i in range(chain_length)],
                    "positions": chain_rel_pos,
                    "orientations": [(1,0,0,0)]*chain_length,
                    "charges": chain_charge,
                    "diameters": [0.0]*chain_length
                    }
                rigid_list += [rigid]

            s.particles.N += mol_dict['N']*chain_length
            s.particles.typeid += mol_dict['N']*chain_id
            s.particles.mass += mol_dict['N']*chain_mass
            s.particles.charge += mol_dict['N']*tdp43_charge
            s.particles.position +=  mol_pos
            s.particles.moment_inertia += [0,0,0]*chain_length*mol_dict['N']
            s.particles.orientation += [(1, 0, 0, 0)]*chain_length*mol_dict['N']
            s.particles.body += [-1]*chain_length*mol_dict['N']
            
            s.bonds.N += len(bond_pairs)
            s.bonds.typeid += [0]*len(bond_pairs)
            s.bonds.group += bond_pairs
            
            n_prev_mol += mol_dict['N']
            n_prev_res += mol_dict['N']*chain_length

        s.particles.N = 
        s.particles.types = aa_type+['R']
        s.particles.typeid = [len(aa_type)] + tdp43_id*n_tdp43s
        s.particles.mass = [ck1d_tot_mass] + tdp43_mass*n_tdp43s
        s.particles.charge = [0] + tdp43_charge*n_tdp43s
        s.particles.position = [positions[-1]] + positions[:-1]
        s.particles.moment_inertia = [I_diag[0], I_diag[1], I_diag[2]] + [0,0,0]*tdp43_length*n_tdp43s 
        s.particles.orientation = [(1, 0, 0, 0)] * (n_tdp43s*tdp43_length+1)
        s.particles.body = [0] + [-1]*tdp43_length*n_tdp43s
        
        rigid.create_bodies(sim.state)
        
        
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
    

def simulate_hps_like(infile):
    # UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
    #        mass -> amu
    #        energy -> kJ/mol
    
    # TIME START
    time_start = time.time()
    
    ### MACROs from file
    ## READ INPUT FILE
    macro_dict = hu.macros_from_infile(infile)
    # Simulation parameters
    production_dt = float(macro_dict['production_dt'])        # Time step for production run in picoseconds
    production_steps = int(macro_dict['production_steps'])                       # Total number of steps 
    production_T = float(macro_dict['production_T'])                      # Temperature for production run in Kelvin
    temp = production_T * 0.00831446                  # Temp is RT [kJ/mol]
    box_length = int(macro_dict['box_lenght'])
    start = int(macro_dict['start'])	                           # 0 -> new simulation, 1 -> restart
    contact_dist = float(macro_dict['contact_dist'])
    Dmu = float(macro_dict['Dmu'])
    seed = int(macro_dict['seed'])
    # Logging time interval
    dt_dump = int(macro_dict['dt_dump'])
    dt_log = int(macro_dict['dt_log'])
    dt_backup = int(macro_dict['dt_backup'])
    dt_try_change = int(macro_dict['dt_try_change'])
    dt_time = int(macro_dict['dt_time'])
    dt_active_ser = int(macro_dict['dt_active_ser'])
    # Files
    stat_file = macro_dict['stat_file']
    file_start = macro_dict['file_start']
    logfile = macro_dict['logfile']
    sysfile = macro_dict['sysfile']
    # Backend
    dev = macro_dict['dev']
    logging_level = macro_dict['logging']

    logging.basicConfig(level=logging_level)

    ## READ stat_file
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
        aa_sigma.append(aa_param_dict[k][2]/10.)
        aa_lambda.append(aa_param_dict[k][3])

    ## READ SYSTEM FILE
    sys_dicts = hu.system_from_file(sysfile)
    
    ck1d_id = hu.chain_id_from_pdb(filein_ck1d, aa_param_dict)
    ck1d_rel_pos = hu.chain_positions_from_pdb(filein_ck1d, relto='com', chain_mass=[aa_mass[ck1d_id[i]] for i in range(ck1d_length)])   # positions relative to c.o.m. 
    
    ### HOOMD3 routine
    ## INITIALIZATION
    if dev=='CPU':
        device = hoomd.device.CPU(notice_level=2)
    elif dev=='GPU':
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



    rigid_mass = snap.particles.mass[0]
    
    type_id = snap.particles.typeid
    ser_serials = np.where(np.isin(type_id[:155],[15,20]))[0]
    activeCK1d_serials = [30800+147, 30800+148, 30800+149]     # [171, 204, 301, 302, 303, 304, 305]
    
    # # rigid body
    rigid = hoomd.md.constrain.Rigid()
    rigid.body['R'] = {
        "constituent_types": [aa_type[ck1d_id[i]] for i in range(ck1d_length)],
        "positions": ck1d_rel_pos,
        "orientations": [(1,0,0,0)]*ck1d_length,
        "charges": [aa_charge[ck1d_id[i]] for i in range(ck1d_length)],
        "diameters": [0.0]*ck1d_length
        }
    
    # # groups
    all_group = hoomd.filter.All()
    moving_group = hoomd.filter.Rigid(("center", "free"))
    
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
    langevin.gamma['R'] = rigid_mass/1000.0
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
    
    changeser_action = ChangeSerine(active_serials=activeCK1d_serials, ser_serials=ser_serials, forces=[yukawa, ashbaugh_table], 
                                    glb_contacts=contacts, temp=temp, Dmu=Dmu, box_size=box_lenght, contact_dist=contact_dist)
    changeser_updater = hoomd.update.CustomUpdater(action=changeser_action, trigger=hoomd.trigger.Periodic(dt_try_change))

    contacts_action = ContactsBackUp(glb_contacts=contacts)
    contacts_bckp_writer = hoomd.write.CustomWriter(action=contacts_action, trigger=hoomd.trigger.Periodic(int(dt_backup/2)))
    
    # ## SET SIMULATION OPERATIONS
    sim.operations.integrator = integrator 
    sim.operations.computes.append(therm_quantities)

    sim.operations.writers.append(dump_gsd)
    sim.operations.writers.append(backup1_gsd)
    sim.operations.writers.append(backup2_gsd)
    sim.operations.writers.append(tq_gsd)
    sim.operations += time_writer
    sim.operations += changeser_updater
    sim.operations += contacts_bckp_writer

    sim.run(production_steps-init_step)
#    sim.run(production_steps)
    if start==1 and len(contacts)!=0:
        cont_prev = np.loadtxt(logfile+"_contacts.txt")
        if len(cont_prev)!=0:
            if cont_prev.ndim==1:
                cont_prev = [cont_prev]
            contacts = np.append(cont_prev, contacts, axis=0)
        np.savetxt(logfile+"_contacts.txt", contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")
    elif start==0:
        np.savetxt(logfile+"_contacts.txt", contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")
    
    hoomd.write.GSD.write(state=sim.state, filename=logfile+'_end.gsd')
    
