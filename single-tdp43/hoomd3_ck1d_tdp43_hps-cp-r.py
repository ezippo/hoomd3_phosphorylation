import sys,os
import time
import numpy as np
import hoomd
import gsd, gsd.hoomd
import logging

import hoomd_util as hu

# --------------------------- MAIN ------------------------------

if __name__=='__main__':
    # TIME START
    time_start = time.time()
    logging.basicConfig(level=logging.DEBUG)

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
    ck1d_pos_arr = np.array(ck1d_pos)/10.
    ck1d_sigma_arr = np.array(ck1d_sigma)/10.
    ck1d_length = len(ck1d_id)       
    ck1d_tot_mass = np.sum(ck1d_mass)   
    ck1d_cof_pos = ( np.sum(ck1d_pos_arr[:,0]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,1]*ck1d_mass)/ck1d_tot_mass , np.sum(ck1d_pos_arr[:,2]*ck1d_mass)/ck1d_tot_mass  )
    ck1d_rel_pos = ck1d_pos_arr - ck1d_cof_pos
    
    # ### HOOMD3 routine
    # ## INITIALIZATION
    device = hoomd.device.CPU(notice_level=2)
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
    
    type_id = snap.particles.typeid
    ser_serials = np.where(np.isin(type_id[:155],[15,20]))[0]
    activeCK1d_serials = [301, 302, 303]     # [171, 204, 301, 302, 303, 304, 305]
 
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
    ser_group = hoomd.filter.Tags(list(ser_serials))
    active_group = hoomd.filter.Tags(activeCK1d_serials)
    active_ser_group = hoomd.filter.Union(active_group, ser_group)
    
    # ## PAIR INTERACTIONS
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond', 'body'))
    
    # # bonds
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params['AA_bond'] = dict(k=8360, r0=0.381)

    def ashbaugh_interactions(types, aa_param_combined, aa2_type_rigid_1, one_rna_type, nl):

        ashbaugh_table = hoomd.md.pair.Table(nlist=nl)
        for i in range(len(aa_type)):
            atom1 = aa_type[i]
            for j in range(i,len(aa_type)):
                atom2 = aa_type[j]
                if ai in aa2_type_rigid or aj in aa2_type_rigid:
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

        nb = hoomd.md.pair.Table(nlist=cell)
        for i, ai in enumerate(types):
            for j, aj in enumerate(types):
                if ai in aa2_type_rigid_1 or ai in aa2_type_rigid_2 or aj in aa2_type_rigid_1 or aj in aa2_type_rigid_2:
                    lam_val = lam_val * 0.7 # reduce lambda value by 30%
                nb.pair_coeff.set(ai, aj, lam=lam_val, epsilon=0.8368,
                                sigma=(aa_param_combined[ai][2] + aa_param_combined[aj][2])/10.0/2.0, r_cut=2.0)
            nb.pair_coeff.set(ai, 'R', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
            nb.pair_coeff.set(ai, 'Z', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
            for ii in one_rna_type:
                for jj in one_rna_type:
                    nb.pair_coeff.set(ii, jj, lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
                nb.pair_coeff.set(ii, 'R', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
                nb.pair_coeff.set(ii, 'Z', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
            for k in one_rna_type:
                nb.pair_coeff.set(ai, k, lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
        nb.pair_coeff.set('R', 'R', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
        nb.pair_coeff.set('R', 'Z', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)
        nb.pair_coeff.set('Z', 'Z', lam=0.0, epsilon=0.0, sigma=0.0, r_cut=0.0)

        return nb
    
    nb = ashbaugh_interactions(types, aa_param_combined, aa2_type_rigid_1, aa2_type_rigid_2, one_rna_type, nl)
