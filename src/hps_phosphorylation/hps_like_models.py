#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import logging
import itertools

import numpy as np
import hoomd
import gsd, gsd.hoomd
import ashbaugh_plugin as aplugin

import hps_phosphorylation.hoomd_util as hu
import hps_phosphorylation.phosphorylation as phospho

### -------------------------------------- PAIR POTENTIALS DEFINITION -------------------------------------------------------------

## YUKAWA: elecrostatic interaction with Debey-Huckel screening
def yukawa_pair_potential(cell, aa_type, R_type_list, aa_charge, model='HPS', temp=300, ionic=0.100, rescale=0):
    yukawa = hoomd.md.pair.Yukawa(nlist=cell)
    yukawa_eps, yukawa_kappa = hu.compute_yukawa_params(temp=temp, ionic=ionic)
    if rescale!=0:
        rigid_types = [f'{name}_r' for name in aa_type]
    for i,atom1 in enumerate(aa_type):
        # interactions IDP-IDP
        for j in range(i,len(aa_type)):             
            atom2 = aa_type[j]
            if model=="CALVADOS2":
                yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*yukawa_eps, kappa=yukawa_kappa)
                yukawa.r_cut[(atom1,atom2)] = 4.0
            else:
                yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*1.73136, kappa=1.0)
                yukawa.r_cut[(atom1,atom2)] = 3.5
            if aa_charge[i]==0 or aa_charge[j]==0:
                yukawa.r_cut[(atom1,atom2)] = 0.0
            logging.debug(f"INTERACTIONS : yukawa {atom1}-{atom2}")
        # interactions IDP-globular
        if rescale!=0:
            for j,atom2 in enumerate(rigid_types):     
                if model=="CALVADOS2":
                    yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*yukawa_eps, kappa=yukawa_kappa)
                    yukawa.r_cut[(atom1,atom2)] = 4.0
                else:
                    yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*1.73136, kappa=1.0)
                    yukawa.r_cut[(atom1,atom2)] = 3.5
                if aa_charge[i]==0 or aa_charge[j]==0:
                    yukawa.r_cut[(atom1,atom2)] = 0.0
                logging.debug(f"INTERACTIONS : yukawa {atom1}-{atom2}")
        # interactions IDP-R particles : no interactions with fictious particles
        for j,atom2 in enumerate(R_type_list):             
            yukawa.params[(atom1,atom2)] = dict(epsilon=0, kappa=1.0)
            yukawa.r_cut[(atom1,atom2)] = 0.0
            logging.debug(f"INTERACTIONS : yukawa {atom1}-{atom2}")
    
    if rescale!=0:
        for i,atom1 in enumerate(rigid_types):
            # interactions globular-globular
            for j in range(i,len(rigid_types)):             
                atom2 = rigid_types[j]
                if model=="CALVADOS2":
                    yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*yukawa_eps, kappa=yukawa_kappa)
                    yukawa.r_cut[(atom1,atom2)] = 4.0
                else:
                    yukawa.params[(atom1,atom2)] = dict(epsilon=aa_charge[i]*aa_charge[j]*1.73136, kappa=1.0)
                    yukawa.r_cut[(atom1,atom2)] = 3.5
                if aa_charge[i]==0 or aa_charge[j]==0:
                    yukawa.r_cut[(atom1,atom2)] = 0.0
                logging.debug(f"INTERACTIONS : yukawa {atom1}-{atom2}")
            # interactions globular-R particles : no interactions with fictious particles
            for j,atom2 in enumerate(R_type_list):             
                yukawa.params[(atom1,atom2)] = dict(epsilon=0, kappa=1.0)
                yukawa.r_cut[(atom1,atom2)] = 0.0
                logging.debug(f"INTERACTIONS : yukawa {atom1}-{atom2}")
    # interactions R-R particles : no interactions between fictious particles        
    for i,atom1 in enumerate(R_type_list):
        for j in range(i,len(R_type_list)):  
            atom2 = R_type_list[j]
            yukawa.params[(atom1,atom2)] = dict(epsilon=0, kappa=1.0)
            yukawa.r_cut[(atom1,atom2)] = 0.0
            logging.debug(f"INTERACTIONS : yukawa {atom1}-{atom2}")

    return yukawa

## ASHBAUGH-HATCH
# ashbaugh_plugin: Van der Waals interactions (with hydrophobicity screening)
def ashbaugh_hatch_pair_potential(cell, aa_type, R_type_list, aa_sigma, aa_lambda, rescale=0):
    ashbaugh = aplugin.pair.AshbaughPair(nlist=cell)
    eps_ashbaugh = 0.8368
    if rescale!=0:
        rigid_types = [f'{name}_r' for name in aa_type]
        r_factor = 1. - rescale/100.
        logging.debug(f"INTERACTIONS: ashbaugh-hatch rescale factor {r_factor}") 
    for i,atom1 in enumerate(aa_type):
        # interactions IDP-IDP
        for j in range(i,len(aa_type)):             
            atom2 = aa_type[j]
            sigma_tmp = (aa_sigma[i] + aa_sigma[j])/2.0
            lam_tmp = (aa_lambda[i] + aa_lambda[j])/2.0
            ashbaugh.params[(atom1, atom2)] = dict(epsilon=eps_ashbaugh, sigma=sigma_tmp, lam=lam_tmp)
            ashbaugh.r_cut[(atom1, atom2)] = 2.0            
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")
        # interactions IDP-globular
        if rescale!=0:
            for j,atom2 in enumerate(rigid_types):     
                sigma_tmp = (aa_sigma[i] + aa_sigma[j])/2.0
                lam_tmp = r_factor*(aa_lambda[i] + aa_lambda[j])/2.0
                ashbaugh.params[(atom1, atom2)] = dict(epsilon=eps_ashbaugh, sigma=sigma_tmp, lam=lam_tmp)
                ashbaugh.r_cut[(atom1, atom2)] = 2.0            
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")          
        # interactions IDP-R particles : no interactions with fictious particles
        for j,atom2 in enumerate(R_type_list):             
            ashbaugh.params[(atom1, atom2)] = dict(epsilon=0, sigma=0, lam=0)
            ashbaugh.r_cut[(atom1, atom2)] = 0 
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      
        
    if rescale!=0:
        for i,atom1 in enumerate(rigid_types):
            # interactions globular-globular
            for j in range(i,len(rigid_types)):             
                atom2 = rigid_types[j]
                sigma_tmp = (aa_sigma[i] + aa_sigma[j])/2.0
                lam_tmp = r_factor*r_factor*(aa_lambda[i] + aa_lambda[j])/2.0
                ashbaugh.params[(atom1, atom2)] = dict(epsilon=eps_ashbaugh, sigma=sigma_tmp, lam=lam_tmp)
                ashbaugh.r_cut[(atom1, atom2)] = 2.0            
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")          
            # interactions globular-R particles : no interactions with fictious particles
            for j,atom2 in enumerate(R_type_list):             
                ashbaugh.params[(atom1, atom2)] = dict(epsilon=0, sigma=0, lam=0)
                ashbaugh.r_cut[(atom1, atom2)] = 0 
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      

    # interactions R-R particles : no interactions between fictious particles        
    for i,atom1 in enumerate(R_type_list):
        for j in range(i,len(R_type_list)):  
            atom2 = R_type_list[j]
            ashbaugh.params[(atom1, atom2)] = dict(epsilon=0, sigma=0, lam=0)
            ashbaugh.r_cut[(atom1, atom2)] = 0 
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      
            
    return ashbaugh


# hoomd3 Lennard-Jones potential: cation-pi interaction (with hydrophobicity screening)
def cation_pi_lj_potential(cell, aa_type, R_type_list, aa_sigma, rescale=0):
    cation_pi_lj = hoomd.md.pair.LJ(nlist=cell)
    cation_type = ["ARG", "LYS"]
    pi_type = ["PHE", "TRP", "TYR"]
    eps_catpi = 3.138
    if rescale!=0:
        rigid_types = [f'{name}_r' for name in aa_type]
        r_factor = 1. - rescale/100.
        logging.debug(f"INTERACTIONS: ashbaugh-hatch rescale factor {r_factor}") 
        cation_type += ["ARG_r", "LYS_r"]
        pi_type += ["PHE_r", "TRP_r", "TYR_r"]
    for i,atom1 in enumerate(aa_type):
        # interactions IDP-IDP
        for j in range(i,len(aa_type)):             
            atom2 = aa_type[j]
            if (atom1 in cation_type and atom2 in pi_type) or (atom2 in cation_type and atom1 in pi_type):
                sigma_tmp = (aa_sigma[i] + aa_sigma[j])/2.0
                cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=eps_catpi, sigma=sigma_tmp)
                cation_pi_lj.r_cut[(atom1, atom2)] = 2.0
            else:
                cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=0, sigma=0)
                cation_pi_lj.r_cut[(atom1, atom2)] = 0
            logging.debug(f"INTERACTIONS: cation-pi {atom1}-{atom2}")
        # interactions IDP-globular
        if rescale!=0:
            for j,atom2 in enumerate(rigid_types): 
                if (atom1 in cation_type and atom2 in pi_type) or (atom2 in cation_type and atom1 in pi_type):
                    sigma_tmp = (aa_sigma[i] + aa_sigma[j])/2.0
                    cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=r_factor*eps_catpi, sigma=sigma_tmp)
                    cation_pi_lj.r_cut[(atom1, atom2)] = 2.0
                else:
                    cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=0, sigma=0)
                    cation_pi_lj.r_cut[(atom1, atom2)] = 0
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")          
        # interactions IDP-R particles : no interactions with fictious particles
        for j,atom2 in enumerate(R_type_list):             
            cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=0, sigma=0)
            cation_pi_lj.r_cut[(atom1, atom2)] = 0 
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      
        
    if rescale!=0:
        for i,atom1 in enumerate(rigid_types):
            # interactions globular-globular
            for j in range(i,len(rigid_types)):             
                atom2 = rigid_types[j]
                if (atom1 in cation_type and atom2 in pi_type) or (atom2 in cation_type and atom1 in pi_type):
                    sigma_tmp = (aa_sigma[i] + aa_sigma[j])/2.0
                    cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=r_factor*r_factor*eps_catpi, sigma=sigma_tmp)
                    cation_pi_lj.r_cut[(atom1, atom2)] = 2.0
                else:
                    cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=0, sigma=0)
                    cation_pi_lj.r_cut[(atom1, atom2)] = 0
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")          
            # interactions globular-R particles : no interactions with fictious particles
            for j,atom2 in enumerate(R_type_list):             
                cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=0, sigma=0)
                cation_pi_lj.r_cut[(atom1, atom2)] = 0 
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      

    # interactions R-R particles : no interactions between fictious particles        
    for i,atom1 in enumerate(R_type_list):
        for j in range(i,len(R_type_list)):  
            atom2 = R_type_list[j]
            cation_pi_lj.params[(atom1, atom2)] = dict(epsilon=0, sigma=0)
            cation_pi_lj.r_cut[(atom1, atom2)] = 0 
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      
            
    return cation_pi_lj


# table potential from hoomd3: Van der Waals interactions + cation-pi interaction (with hydrophobicity screening) 
def table_ashbaugh_pair_potential(cell, aa_type, R_type_list, aa_sigma, aa_lambda, model='HPS', rescale=0):
    ashbaugh_table = hoomd.md.pair.Table(nlist=cell)
    cation_type = ["ARG", "LYS"]
    pi_type = ["PHE", "TRP", "TYR"]
    if rescale!=0:
        rigid_types = [f'{name}_r' for name in aa_type]
        r_factor = 1. - rescale/100.
        logging.debug(f"INTERACTIONS: ashbaugh-hatch rescale factor {r_factor}") 
        cation_type += ["ARG_r", "LYS_r"]
        pi_type += ["PHE_r", "TRP_r", "TYR_r"]
    for i,atom1 in enumerate(aa_type):
        # interactions IDP-IDP
        for j in range(i,len(aa_type)):             
            atom2 = aa_type[j]
            Ulist = np.array(hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                    lambda_hps=[aa_lambda[i], aa_lambda[j]],
                                    r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368) )
            Flist = np.array(hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                    lambda_hps=[aa_lambda[i], aa_lambda[j]],
                                    r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368) )
            if model=="HPS_cp":
                # cation-pi interaction
                if (atom1 in cation_type and atom2 in pi_type) or (atom2 in cation_type and atom1 in pi_type):
                    Ulist += np.array(hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                            lambda_hps=1.0,
                                            r_max=2.0, r_min=0.2, n_bins=100000, epsilon=3.138) )
                    Flist += np.array(hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                            lambda_hps=1.0,
                                            r_max=2.0, r_min=0.2, n_bins=100000, epsilon=3.138) )
            ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0.2, U=Ulist, F=Flist)
            ashbaugh_table.r_cut[(atom1, atom2)] = 2.0            
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")
        # interactions IDP-globular
        if rescale!=0:
            for j,atom2 in enumerate(rigid_types):     
                Ulist = np.array(hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                        lambda_hps=[r_factor*aa_lambda[i], r_factor*aa_lambda[j]],                # Scale down lamba by rescale% for interactions with rigid body particles 
                                        r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368) )
                Flist = np.array(hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                        lambda_hps=[r_factor*aa_lambda[i], r_factor*aa_lambda[j]],
                                        r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368) )
                if model=="HPS_cp":
                    # cation-pi interaction
                    if (atom1 in cation_type and atom2 in pi_type) or (atom2 in cation_type and atom1 in pi_type):
                        Ulist += np.array(hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                                lambda_hps=1.0,
                                                r_max=2.0, r_min=0.2, n_bins=100000, epsilon=r_factor*3.138) )         # Scale down epsilon by rescale% for interactions with rigid body particles 
                        Flist += np.array(hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                                lambda_hps=1.0,
                                                r_max=2.0, r_min=0.2, n_bins=100000, epsilon=r_factor*3.138) )
                ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0.2, U=Ulist, F=Flist)
                ashbaugh_table.r_cut[(atom1, atom2)] = 2.0            
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")          
        # interactions IDP-R particles : no interactions with fictious particles
        for j,atom2 in enumerate(R_type_list):             
            ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0., U=[0], F=[0])
            ashbaugh_table.r_cut[(atom1, atom2)] = 0 
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      
        
    if rescale!=0:
        for i,atom1 in enumerate(rigid_types):
            # interactions globular-globular
            for j in range(i,len(rigid_types)):             
                atom2 = rigid_types[j]
                Ulist = np.array(hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                        lambda_hps=[r_factor*r_factor*aa_lambda[i], r_factor*r_factor*aa_lambda[j]],                # Scale down lamba by rescale*rescale% for interactions between rigid body particles 
                                        r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368) )
                Flist = np.array(hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                        lambda_hps=[r_factor*r_factor*aa_lambda[i], r_factor*r_factor*aa_lambda[j]],
                                        r_max=2.0, r_min=0.2, n_bins=100000, epsilon=0.8368) )
                if model=="HPS_cp":
                    # cation-pi interaction
                    if (atom1 in cation_type and atom2 in pi_type) or (atom2 in cation_type and atom1 in pi_type):
                        Ulist += np.array(hu.Ulist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                                lambda_hps=1.0,
                                                r_max=2.0, r_min=0.2, n_bins=100000, epsilon=r_factor*r_factor*3.138) )         # Scale down epsilon by rescale*rescale% for interactions between rigid body particles 
                        Flist += np.array(hu.Flist_ashbaugh(sigma=[aa_sigma[i], aa_sigma[j]], 
                                                lambda_hps=1.0,
                                                r_max=2.0, r_min=0.2, n_bins=100000, epsilon=r_factor*r_factor*3.138) )
                ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0.2, U=Ulist, F=Flist)
                ashbaugh_table.r_cut[(atom1, atom2)] = 2.0            
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")          
            # interactions globular-R particles : no interactions with fictious particles
            for j,atom2 in enumerate(R_type_list):             
                ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0., U=[0], F=[0])
                ashbaugh_table.r_cut[(atom1, atom2)] = 0 
                logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      

    # interactions R-R particles : no interactions between fictious particles        
    for i,atom1 in enumerate(R_type_list):
        for j in range(i,len(R_type_list)):  
            atom2 = R_type_list[j]
            ashbaugh_table.params[(atom1, atom2)] = dict(r_min=0., U=[0], F=[0])
            ashbaugh_table.r_cut[(atom1, atom2)] = 0 
            logging.debug(f"INTERACTIONS: ashbaugh-hatch {atom1}-{atom2}")      
            
    return ashbaugh_table



### --------------------------------- CREATE INITIAL CONFIGURATION MODE ------------------------------------------------

def create_init_configuration(filename, syslist, aa_param_dict, box_length, rescale=False):
    n_mols = len(syslist)
    n_chains = np.sum([int(syslist[i]['N']) for i in range(n_mols)])
    aa_type = list(aa_param_dict.keys())
    if rescale:
        aa_type_r = [f'{aa_name}_r' for aa_name in aa_type]
    
    ## initialize first snapshot
    s=gsd.hoomd.Frame()
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0
    s.particles.N = 0
    s.particles.types = []
    s.particles.types += aa_type
    s.particles.typeid = []
    s.particles.mass = []
    s.particles.charge = []
    s.particles.position = []
    s.particles.moment_inertia = [] 
    s.particles.orientation = []
    #s.particles.body = []
    s.bonds.N = 0
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = []
    s.bonds.group = []

    ## create array of c.o.m positions for all the molecules
    K = math.ceil(n_chains**(1/3))
    spacing = box_length/K
    x = np.linspace(-box_length/2, box_length/2, K, endpoint=False)
    positions = list(itertools.product(x, repeat=3))
    positions = np.array(positions) + [spacing/2, spacing/2, spacing/2]
    np.random.shuffle(positions)
    positions = positions[:n_chains, :]
    
    ### LOOP ON THE MOLECULES TYPE
    rigid = hoomd.md.constrain.Rigid()
    n_prev_mol = 0
    n_prev_res = 0
    chain_lengths_list = []
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        chain_id, chain_mass, chain_charge, _, _ = hu.aa_stats_sequence(mol_dict['pdb'], aa_param_dict)
        chain_length = len(chain_id)
        chain_lengths_list += [chain_length]
        chain_rel_pos = hu.chain_positions_from_pdb(mol_dict['pdb'], relto='com', chain_mass=chain_mass)   # positions relative to c.o.m. 
        n_mol_chains = int(mol_dict['N'])
            
        ## intrinsically disordered case
        if mol_dict['rigid']=='0':
            mol_pos = []
            bond_pairs=[]
            for i_chain in range(n_mol_chains):
                mol_pos += list(chain_rel_pos+positions[n_prev_mol+i_chain])
                bond_pairs += [[n_prev_res + i+i_chain*chain_length, n_prev_res + i+1+i_chain*chain_length] for i in range(chain_length-1)]

            s.particles.N += n_mol_chains*chain_length
            s.particles.typeid += n_mol_chains*chain_id
            s.particles.mass += n_mol_chains*chain_mass
            s.particles.charge += n_mol_chains*chain_charge
            s.particles.position +=  mol_pos
            s.particles.moment_inertia += [[0,0,0]]*chain_length*n_mol_chains
            s.particles.orientation += [(1, 0, 0, 0)]*chain_length*n_mol_chains
            
            s.bonds.N += len(bond_pairs)
            s.bonds.typeid += [0]*len(bond_pairs)
            s.bonds.group += bond_pairs

            n_prev_res += chain_length*n_mol_chains
            
        ## case with rigid bodies
        else:
            rigid_ind_l = hu.read_rigid_indexes(mol_dict['rigid'])
            n_rigids = len(rigid_ind_l)

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
            count_res=0
            # loop on the rigid bodies of the same molecule
            for r in range(n_rigids):
                # rigid body and previous free body indexes of the chain
                rigid_ind = rigid_ind_l[r]
                free_ind = [i for i in range(count_res, rigid_ind[0])]
                # rigid body properties
                types_rigid_bodies += ['R'+str(len(s.particles.types)-len(aa_type)+r+1)]                        # type R1, R2, R3 ...
                typeid_rigid_bodies += [len(s.particles.types)+len(types_rigid_bodies)-1]  
                rigid_mass = [chain_mass[i] for i in rigid_ind]             
                mass_rigid_bodies += [np.sum(rigid_mass)]                   # total mass of the rigid body
                rigid_rel_pos = chain_rel_pos[rigid_ind] 
                reshaped_rigid_mass = np.reshape( rigid_mass, (len(rigid_mass),1) )
                rigid_com_rel_pos = np.sum(rigid_rel_pos * reshaped_rigid_mass, axis=0) / np.sum(rigid_mass)       # c.o.m. relative to the center of the molecule
                rigid_rel_pos = rigid_rel_pos-rigid_com_rel_pos             # positions of monomers of the rigid body relative to the c.o.m.
                position_rigid_bodies += [rigid_com_rel_pos]   
                I = hu.protein_moment_inertia(rigid_rel_pos, rigid_mass)
                I_diag, E_vec = np.linalg.eig(I)
                moment_inertia_rigid_bodies += [[I_diag[0], I_diag[1], I_diag[2]]]
                # create rigid body object
                if rescale:
                    rigid.body[types_rigid_bodies[-1]] = {
                        "constituent_types": [aa_type_r[chain_id[i]] for i in rigid_ind],
                        "positions": rigid_rel_pos,
                        "orientations": [(1,0,0,0)]*len(rigid_ind),
                        "charges": [ chain_charge[i] for i in rigid_ind ],
                        "diameters": [0.0]*len(rigid_ind)
                        }
                else:
                    rigid.body[types_rigid_bodies[-1]] = {
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
                    position_free_bodies += list(free_rel_pos)     
                    length_free_bodies += [len(free_ind)]
                        
                count_res += len(free_ind) + len(rigid_ind)
                
            # free monomers in the final tail
            if rigid_ind[-1]<len(chain_id)-1:
                free_ind = [i for i in range(count_res, len(chain_id))]
                typeid_free_bodies += [chain_id[i] for i in free_ind]
                mass_free_bodies += [chain_mass[i] for i in free_ind]
                charge_free_bodies += [chain_charge[i] for i in free_ind]
                free_rel_pos = chain_rel_pos[free_ind]                      # positions of the free monomers relative to the center of the molecule
                position_free_bodies += list(free_rel_pos)  
                length_free_bodies += [len(free_ind)]
            else:
                length_free_bodies += [0]

            length_free_total = np.sum(length_free_bodies)

            # shift positions to the c.o.m of the molecule
            all_positions_rel = position_rigid_bodies+position_free_bodies
            all_positions = []
            for nm in range(n_mol_chains):
                all_positions += list( np.array(all_positions_rel) + positions[n_prev_mol+nm] )

            # bonds between free monomers
            bond_pairs = []
            for nc in range(n_mol_chains):
                for ifree in range(n_rigids+1):
                    bond_pairs += [[n_prev_res + nc*(n_rigids+length_free_total) +n_rigids+np.sum(length_free_bodies[:ifree], dtype=int)+i , n_prev_res + nc*(n_rigids+length_free_total) +n_rigids+np.sum(length_free_bodies[:ifree], dtype=int)+i+1 ] for i in range(length_free_bodies[ifree]-1)]

            ## add R and free to the snapshot
            # particles
            s.particles.N += n_mol_chains*(n_rigids + length_free_total )
            s.particles.types += types_rigid_bodies
            s.particles.typeid += n_mol_chains*(typeid_rigid_bodies + typeid_free_bodies)
            s.particles.mass += n_mol_chains*(mass_rigid_bodies + mass_free_bodies)
            s.particles.charge += n_mol_chains*([0]*n_rigids + charge_free_bodies)
            s.particles.position += all_positions
            s.particles.moment_inertia += n_mol_chains*(moment_inertia_rigid_bodies + [[0,0,0]]*length_free_total )
            s.particles.orientation += n_mol_chains*([(1, 0, 0, 0)]*(n_rigids+length_free_total) )
            # bonds
            s.bonds.N += len(bond_pairs)
            s.bonds.typeid += [0]*len(bond_pairs)
            s.bonds.group += bond_pairs
        
            n_prev_res += (n_rigids+length_free_total)*n_mol_chains

        n_prev_mol += n_mol_chains
    
    bond_pairs_tot = s.bonds.group

    ### BUILD RIGID BODIES
    if rescale:
        s.particles.types += aa_type_r
    
    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_snapshot(s)
    rigid.create_bodies(sim.state)
    integrator = hoomd.md.Integrator(dt=0.01, integrate_rotational_dof=True)
    integrator.rigid = rigid
    sim.operations.integrator = integrator
    sim.run(0)
 
    hoomd.write.GSD.write(state=sim.state, filename=filename, mode='wb')

    ### ADD BONDS FREE-RIGID 
    s1 = gsd.hoomd.open(name=filename, mode='r+')[0]
    
    ## indexing
    reordered_ind = hu.reordering_index(syslist)

    ## bonds free rigid
    bonds_free_rigid = []
    n_prev_res = 0
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        if mol_dict['rigid']!='0':
            rigid_ind_l = hu.read_rigid_indexes(mol_dict['rigid'])
            n_rigids = len(rigid_ind_l)
            for ch in range(int(mol_dict['N'])):
                for nr in range(n_rigids):
                    start_rig = n_prev_res + rigid_ind_l[nr][0]
                    if start_rig > n_prev_res:
                        bonds_free_rigid += [[reordered_ind[n_rigids+start_rig-1], reordered_ind[n_rigids+start_rig]]]
                    end_rig = n_prev_res + rigid_ind_l[nr][-1]
                    if end_rig < n_prev_res + chain_lengths_list[mol]-1:
                        bonds_free_rigid += [[reordered_ind[n_rigids+end_rig], reordered_ind[n_rigids+end_rig+1]]]
                n_prev_res += chain_lengths_list[mol] + n_rigids
        else:
            n_prev_res += chain_lengths_list[mol]*int(mol_dict['N'])

    bond_pairs_tot += bonds_free_rigid

    s1.bonds.N = len(bond_pairs_tot) 
    s1.bonds.typeid = [0]*len(bond_pairs_tot)
    s1.bonds.group = bond_pairs_tot
    print(s1.particles.N)
    with gsd.hoomd.open(name=filename, mode='w') as fout:
        fout.append(s1)
        fout.close()


### --------------------------------- SIMULATION MODE ------------------------------------------------

def simulate_hps_like(macro_dict, aa_param_dict, syslist, model='HPS', rescale=0, mode='relax', resize=None):
    # UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
    #        mass -> amu
    #        energy -> kJ/mol
    
    # TIME START
    time_start = time.time()
    
    ### MACROs from file
    # Simulation parameters
    production_dt = float(macro_dict['production_dt'])        # Time step for production run in picoseconds
    production_steps = int(macro_dict['production_steps'])                       # Total number of steps 
    production_T = float(macro_dict['production_T'])                      # Temperature for production run in Kelvin
    temp = production_T * 0.00831446                  # Temp is RT [kJ/mol]
    ionic = float(macro_dict['ionic'])          # ionic strength [M], usually around 0.100-0.150 M
    start = int(macro_dict['start'])	                           # 0 -> new simulation, 1 -> restart
    if isinstance(macro_dict['box'], str):
        box_size = [float(macro_dict['box'])]*3
    else:
        box_size = list(macro_dict['box'])              # box side lengths [Lx Ly Lz]
        box_size = [ float(box_size[i]) for i in range(3) ]

    seed = int(macro_dict['seed'])
    # Logging time interval
    dt_dump = int(macro_dict['dt_dump'])
    dt_log = int(macro_dict['dt_log'])
    dt_backup = int(macro_dict['dt_backup'])
    dt_time = int(macro_dict['dt_time'])
    
    # Files
    file_start = macro_dict['file_start']
    logfile = macro_dict['logfile']
    # Backend
    dev = macro_dict['dev']
    logging_level = macro_dict['logging']
    logging.basicConfig(level=logging_level)
    
    logging.debug(f"INPUT : macro_dict: {macro_dict}")

    # Input parameters for all the amino acids 
    aa_type = list(aa_param_dict.keys())
    logging.debug(f"INPUT : aa_type: {aa_type}")
    aa_mass = []
    aa_charge = []
    aa_sigma = []
    aa_lambda = []
    for k in aa_type:
        aa_mass.append(aa_param_dict[k][0])
        aa_charge.append(aa_param_dict[k][1])
        aa_sigma.append(aa_param_dict[k][2]/10.)
        aa_lambda.append(aa_param_dict[k][3])
    if rescale!=0:
        aa_type_r = [f"{name}_r" for name in aa_type]
        logging.debug(f"INPUT : aa_type rigids: {aa_type_r}")
    
    # molecules chain lengths
    n_mols = len(syslist)
    chain_lengths_l = []
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        chain_id = hu.chain_id_from_pdb(mol_dict['pdb'], aa_param_dict)
        chain_lengths_l += [len(chain_id)]
    logging.debug(f"INPUT : chain lengths: {chain_lengths_l}")


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

    # indexing and types
    type_id = snap.particles.typeid
    logging.debug(snap.particles.types)
    
    # rigid bodies 
    rigid, rigid_masses_l, n_rigids_l, R_type_list = hu.rigidbodies_from_syslist(syslist, chain_lengths_l, aa_param_dict, rescale)
    logging.debug(f"RIGID : rigid names: {R_type_list}")
    logging.debug(f"RIGID : n_rigids_l: {n_rigids_l}")
    
    # phosphosite
    ser_serials = phospho.phosphosites_from_syslist(syslist, type_id, chain_lengths_l, n_rigids_l)
    logging.debug(f"PHOSPHOSITES : ser_serials: {ser_serials}")

    # active site
    active_serials_l = phospho.activesites_from_syslist(syslist, chain_lengths_l, n_rigids_l)
    logging.debug(f"ACTIVE SITES : active_serials list: {active_serials_l}")

    
    # groups
    all_group = hoomd.filter.All()
    moving_group = hoomd.filter.Rigid(("center", "free"))
    
    ## PAIR INTERACTIONS
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond', 'body'))
    
    # bonds
    harmonic = hoomd.md.bond.Harmonic()
    if model=="CALVADOS2":
        harmonic.params['AA_bond'] = dict(k=8033, r0=0.381)
    else:
        harmonic.params['AA_bond'] = dict(k=8360, r0=0.381)
        
    # electrostatics forces
    yukawa = yukawa_pair_potential(cell, aa_type, R_type_list, aa_charge, temp, ionic, model, rescale)
    
    # nonbonded: ashbaugh-hatch potential
    # ashbaugh_table = ashbaugh_hatch_pair_potential(cell, aa_type, R_type_list, aa_sigma, aa_lambda, model, rescale)
    # logging.debug(f"POTENTIALS : yukawa pair potential: {yukawa}")
    ashbaugh = ashbaugh_hatch_pair_potential(cell, aa_type, R_type_list, aa_sigma, aa_lambda, rescale)
    if model=='HPS_cp':
        cationpi_lj = cation_pi_lj_potential(cell, aa_type, R_type_list, aa_sigma, rescale)

    # ## INTEGRATOR
    integrator = hoomd.md.Integrator(production_dt, integrate_rotational_dof=True)        
    
    # method : Langevin
    langevin = hoomd.md.methods.Langevin(filter=moving_group, kT=temp)
    for i,name in enumerate(aa_type):
        langevin.gamma[name] = aa_mass[i]/1000.0
        langevin.gamma_r[name] = (0.0, 0.0, 0.0)
    if rescale!=0:
        for i,name in enumerate(aa_type_r):
            langevin.gamma[name] = aa_mass[i]/1000.0
            langevin.gamma_r[name] = (0.0, 0.0, 0.0)
    for i in range( len(rigid_masses_l) ):
        langevin.gamma['R'+str(i+1)] = rigid_masses_l[i]/1000.0
        langevin.gamma_r['R'+str(i+1)] = (4.0, 4.0, 4.0)
        
    # constraints : rigid body
    integrator.rigid = rigid
    
    # forces 
    integrator.forces.append(harmonic)
    integrator.forces.append(yukawa)
    # integrator.forces.append(ashbaugh_table)
    integrator.forces.append(ashbaugh)
    if model=='HPS_cp':
        integrator.forces.append(cationpi_lj)
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
    time_action = hu.PrintTimestep(time_start, production_steps)
    time_writer = hoomd.write.CustomWriter(action=time_action, trigger=hoomd.trigger.Periodic(dt_time))
    
    # ### if there are no active sites, we don't need to check distances or have phosphorylations
    if len(active_serials_l)!=0:
        contact_dist = float(macro_dict['contact_dist'])
        dt_try_change = int(macro_dict['dt_try_change'])
        contacts = []               # initialize contacts list

        if mode == 'nophospho':
            detector_actions_l = []
            detector_updaters_l = []
            for i,active_serial in enumerate(active_serials_l):
                detector_actions_l += [ phospho.ContactDetector(active_serials=active_serial, ser_serials=ser_serials, glb_contacts=contacts,
                                            box_size=box_size, contact_dist=contact_dist) ]
                detector_updaters_l += [ hoomd.update.CustomUpdater(action=detector_actions_l[-1], trigger=hoomd.trigger.Periodic(dt_try_change)) ]
                
        else:
            changeser_actions_l = []
            changeser_updaters_l = []
            Dmu_array = macro_dict['Dmu']
            if isinstance(Dmu_array, str):
                Dmu_array = [Dmu_array]
            if len(Dmu_array) != len(active_serials_l):
                print('ERROR: parameter Dmu in input file must have a number of input equal to the number of enzymes present in the simulation! ')
                exit()

            if mode == 'relax':
                for i,active_serial in enumerate(active_serials_l):
                    if model=='HPS_cp':
                        changeser_actions_l += [ phospho.ChangeSerine(active_serials=active_serial, ser_serials=ser_serials, forces=[yukawa, ashbaugh, cationpi_lj], 
                                                glb_contacts=contacts, temp=temp, Dmu=float(Dmu_array[i]), box_size=box_size, contact_dist=contact_dist, enzyme_ind=i) ]
                    else:
                        changeser_actions_l += [ phospho.ChangeSerine(active_serials=active_serial, ser_serials=ser_serials, forces=[yukawa, ashbaugh], 
                                                glb_contacts=contacts, temp=temp, Dmu=float(Dmu_array[i]), box_size=box_size, contact_dist=contact_dist, enzyme_ind=i) ]
                    changeser_updaters_l += [ hoomd.update.CustomUpdater(action=changeser_actions_l[-1], trigger=hoomd.trigger.Periodic(dt_try_change, phase=i)) ]
                    
            if mode == 'ness':
                bath_dist = float(macro_dict['bath_dist'])
                dt_bath = int(macro_dict['dt_bath'])
                changes = []
                bath_actions_l = []
                bath_updaters_l = []

                for i,active_serial in enumerate(active_serials_l):
                    if model=='HPS_cp':
                        changeser_actions_l += [ phospho.ChangeSerineNESS(active_serials=active_serial, ser_serials=ser_serials, forces=[yukawa, ashbaugh, cationpi_lj], 
                                            glb_contacts=contacts, glb_changes=changes, temp=temp, Dmu=float(Dmu_array[i]), box_size=box_size, contact_dist=contact_dist) ]
                    else:
                        changeser_actions_l += [ phospho.ChangeSerineNESS(active_serials=active_serial, ser_serials=ser_serials, forces=[yukawa, ashbaugh], 
                                            glb_contacts=contacts, glb_changes=changes, temp=temp, Dmu=float(Dmu_array[i]), box_size=box_size, contact_dist=contact_dist) ]
                    changeser_updaters_l += [ hoomd.update.CustomUpdater(action=changeser_actions_l[-1], trigger=hoomd.trigger.Periodic(dt_try_change, phase=i)) ]

                for i,active_serial in enumerate(active_serials_l):
                    if model=='HPS_cp':
                        bath_actions_l += [ phospho.ReservoirExchange(active_serials=active_serial, ser_serials=ser_serials, forces=[yukawa, ashbaugh, cationpi_lj], 
                                            glb_changes=changes, temp=temp, Dmu=float(Dmu_array[i]), box_size=box_size, bath_dist=bath_dist) ]
                    else:
                        bath_actions_l += [ phospho.ReservoirExchange(active_serials=active_serial, ser_serials=ser_serials, forces=[yukawa, ashbaugh], 
                                            glb_changes=changes, temp=temp, Dmu=float(Dmu_array[i]), box_size=box_size, bath_dist=bath_dist) ]
                    bath_updaters_l += [ hoomd.update.CustomUpdater(action=bath_actions_l[-1], trigger=hoomd.trigger.Periodic(dt_bath, phase=i)) ]
            
                changes_action = phospho.ChangesBackUp(glb_changes=changes, logfile=logfile)
                changes_bckp_writer = hoomd.write.CustomWriter(action=changes_action, trigger=hoomd.trigger.Periodic(int(dt_backup/2)))
        
        contacts_action = phospho.ContactsBackUp(glb_contacts=contacts, logfile=logfile)
        contacts_bckp_writer = hoomd.write.CustomWriter(action=contacts_action, trigger=hoomd.trigger.Periodic(int(dt_backup/2)))
    
    # # Box resize
    if resize != None:
        ramp = hoomd.variant.Ramp(A=0, B=1, t_start=init_step, t_ramp=production_steps-init_step)
        initial_box = sim.state.box
        final_box = hoomd.Box(Lx=resize[0], Ly=resize[1], Lz=resize[2])
        box_resize_trigger = hoomd.trigger.Periodic(10)
        box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)

        sim.operations.updaters.append(box_resize)

    # ## SET SIMULATION OPERATIONS
    sim.operations.integrator = integrator 
    sim.operations.computes.append(therm_quantities)

    sim.operations.writers.append(dump_gsd)
    sim.operations.writers.append(backup1_gsd)
    sim.operations.writers.append(backup2_gsd)
    sim.operations.writers.append(tq_gsd)
    sim.operations += time_writer
    if len(active_serials_l)!=0:
        if mode == 'nophospho':
            for i in range(len(active_serials_l)):
                sim.operations += detector_updaters_l[i]
        else:    
            for i in range(len(active_serials_l)):
                sim.operations += changeser_updaters_l[i]
            if mode == 'ness':
                for i in range(len(active_serials_l)):
                    sim.operations += bath_updaters_l[i]
                sim.operations += changes_bckp_writer
        sim.operations += contacts_bckp_writer
    
    sim.run(production_steps-init_step)

    # ## save contacts list
    if len(active_serials_l)!=0:

        if start==1 and len(contacts)!=0:
            cont_prev = np.loadtxt(logfile+"_contacts.txt")
            if len(cont_prev)!=0:
                if cont_prev.ndim==1:
                    cont_prev = [cont_prev]
                contacts = np.append(cont_prev, contacts, axis=0)
            np.savetxt(logfile+"_contacts.txt", contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")
        elif start==0:
            np.savetxt(logfile+"_contacts.txt", contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")
        
        if mode == 'ness':
            if start==1 and len(changes)!=0:
                cont_prev = np.loadtxt(logfile+"_changes.txt")
                if len(cont_prev)!=0:
                    if cont_prev.ndim==1:
                        cont_prev = [cont_prev]
                    changes = np.append(cont_prev, changes, axis=0)
                np.savetxt(logfile+"_changes.txt", changes, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {1->phosphorylation, 10->change SER with SEP, -1->dephospho accepted, -10->change SEP with SER} ")
            elif start==0:
                np.savetxt(logfile+"_changes.txt", changes, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {1->phosphorylation, 10->change SER with SEP, -1->dephospho accepted, -10->change SEP with SER} ")
    
    hoomd.write.GSD.write(state=sim.state, filename=logfile+'_end.gsd')


if __name__=='__main__':
    infile = 'examples/sim_try/input_try.in'
    macro_dict = hu.macros_from_infile(infile)
    aa_param_dict = hu.aa_stats_from_file(macro_dict['stat_file'])
    syslist = hu.system_from_file(macro_dict['sysfile'])
    reord = hu.reordering_index(syslist)
    print(reord[:300])
    simulate_hps_like(macro_dict, aa_param_dict, syslist, model='HPS_cp', rescale=30)


