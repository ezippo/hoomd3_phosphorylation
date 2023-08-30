#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
import itertools
import math

import numpy as np
import gsd, gsd.hoomd
import hoomd

# UNITS: distance -> nm   (!!!positions and sigma in files are in agstrom!!!)
#        mass -> amu
#        energy -> kJ/mol
# ### MACROs

def aa_stats_from_file(filename):
    '''
    Parameters
    ----------
    filename : str
        name of stats file.

    Returns
    -------
    aa_dict : dicct
        dict('amino acid name':[mass, charge, sigma, lambda])
    '''
    aa_dict = {}
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]!='#':
                line_list = line.rsplit()
                aa_dict[line_list[0]] = np.loadtxt(line_list[1:], dtype=float)
    return aa_dict

def aa_stats_sequence(filename, aa_dict):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    aa_dict : dict
        dict('amino acid name':[mass, charge, sigma, lambda]).

    Returns
    -------
    chain_id : list
        List of a.a. id numbers of the sequence.
    chain_mass : list
        List of a.a. masses of the sequence.
    chain_charge : list
        List of a.a. charges of the sequence.
    chain_sigma : list
        List of a.a. radia of the sequence.
    chain_pos : list
        List of a.a. position tuple (x,y,z) of the sequence.
    '''
    chain_id = []
    chain_mass = []
    chain_charge = []
    chain_sigma = []
    chain_pos = []
    aa_keys = list(aa_dict.keys()) 
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                aa_name = line_list[3]
                chain_id.append(aa_keys.index(aa_name))
                chain_mass.append(aa_dict[aa_name][0])
                chain_charge.append(aa_dict[aa_name][1])
                chain_sigma.append(aa_dict[aa_name][2])
                chain_pos.append( (float(line_list[6]), float(line_list[7]), float(line_list[8])) )
    return chain_id, chain_mass, chain_charge, chain_sigma, chain_pos


def chain_id_from_pdb(filename, aa_dict):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    aa_dict : dict
        dict('amino acid name':[mass, charge, sigma, lambda]).

    Returns
    -------
    chain_id : list
        List of a.a. id numbers of the sequence.
    '''
    chain_id = []
    aa_keys = list(aa_dict.keys()) 
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                aa_name = line_list[3]
                chain_id.append(aa_keys.index(aa_name))
    return chain_id


def chain_positions_from_pdb(filename, relto=None, chain_mass=None, unit='nm'):
    '''
    Parameters
    ----------
    filename : str
        Name of the file with the chain sequence.
    relto : str (default None)
        if None: extracts raw positions from pdb file;
        if 'com': computes positions relative to the center-of-mass. In this case, it is mandatory to specify the parameter 'chain_mass';
        if 'cog': computes positions relative to the center-of-geometry.
    chain_mass : list (default None)
        if relto='com', you need to specify the list of a.a. masses of the sequence;
    unit : str (default 'nm')
        if 'nm': devides the postitions values by 10
        if 'A': keeps the values in Angstrom

    Returns
    -------
    chain_pos : ndarray
        Numpay array of a.a. positions (x,y,z) of the sequence.
    '''
    chain_pos_l = []
    with open(filename, 'r') as fid:
        for line in fid:
            if line[0]=='A':
                line_list = line.rsplit()
                chain_pos_l.append( (float(line_list[6]), float(line_list[7]), float(line_list[8])) )
    if unit=='A':
        chain_pos = np.array(chain_pos_l)
    elif unit=='nm':
        chain_pos = np.array(chain_pos_l)/10.

    if relto==None:
        return chain_pos
    elif relto=='cog':
        return chain_pos - np.mean(chain_pos, axis=0)
    elif relto=='com':
        reshaped_mass = np.reshape( chain_mass, (len(chain_mass),1) )
        chain_com_pos = np.sum(chain_pos * reshaped_mass, axis=0) / np.sum(chain_mass)
        return chain_pos - chain_com_pos
    else:
        print("ERROR: relto option can only be None, 'cog' or 'com'. The insterted value is not valid! ")
        exit()

def system_from_file(filename):
    '''
    Parameters
    ----------
    filename : str
        name of sysfile.

    Returns
    -------
    dict_list : list of dicts
        [ dict('mol1': ['pdb1', 'N1', 'rigid1', 'active_sites1', 'phospho_sites1']), 
          dict('mol2': ['pdb2', 'N2', 'rigid2', 'active_sites2', 'phospho_sites2']),
          ... ]
    '''
    dict_list = []
    with open(filename, 'r') as fid:
        for line in fid:
            mol_dict = dict()              
            if not line.startswith("#") and not line.isspace():
                line_list = np.array(line.rsplit())
                mol_dict['mol'] = line_list[0]    # mol name 
                mol_dict['pdb'] = line_list[1]    # pdb file 
                mol_dict['N'] = line_list[2]      # N molecules 
                mol_dict['rigid'] = line_list[3]    # rigid body indexes
                mol_dict['active_sites'] = line_list[4]    # active site indexes
                mol_dict['phospho_sites'] = line_list[5]    # phospho site indexes
                dict_list += [mol_dict]
                
    return dict_list

def read_rigid_indexes(rigid_str):
    rigid_list = []
    if rigid_str=='0':
        return rigid_list
    else:
        rigid_bodies = rigid_str.rsplit(',')
        for body in rigid_bodies:
            init_body, end_body = np.array(body.rsplit('-'), dtype=int)
            rigid_list += [ np.linspace(init_body-1, end_body-1, end_body-init_body+1, endpoint=True, dtype=int) ]
        return rigid_list

def protein_moment_inertia(chain_rel_pos, chain_mass, chain_sigma=None):
    '''
    Parameters
    ----------
    chain_rel_pos : list
        List of a.a. position tuple (x,y,z) of the sequence.
    chain_mass : list
        List of a.a. masses of the sequence.
    chain_sigma : list, optional
        List of a.a. radia of the sequence.

    Returns
    -------
    I : array
        Moment of inertia tensor.
    '''
    I = np.zeros((3,3))
    if chain_sigma==None:      # particle is a point
        for i,r in enumerate(chain_rel_pos):
            I += chain_mass[i]*( np.dot(r,r)*np.identity(3) - np.outer(r, r) )
    else:                      # particle is a sphere
        for i,r in enumerate(chain_rel_pos):
            I_ref = 2 / 5 * chain_mass[i]*chain_sigma[i]*chain_sigma[i]*np.identity(3)
            I += I_ref + ck1d_mass[i]*( np.dot(r,r)*np.identity(3) - np.outer(r, r) )
    return I


def rigid_dict_from_syslist(syslist):
    n_mols = len(syslist)
    mol_keys = [syslist[mol]['mol'] for mol in range(n_mols)]
    rigid_dict = dict()
    for mol in range(n_mols):
        key = mol_keys[mol]
        mol_dict = syslist[mol]
        chain_length = 0
        with open(mol_dict['pdb'], 'r') as fid:
            for line in fid:
                if line[0]=='A':
                    chain_length += 1

        if mol_dict['rigid']=='0':
            rigid_dict[key] = {
                                "n_rigids": 0, 
                                "rigid_lengths": [], 
                                "free_lengths": [chain_length],
                                "n_chains": int(mol_dict['N'])
                                }
        else:
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
            n_rigids = len(rigid_ind_l)
            rigid_lengths = [ len(rigid_ind_l[nr]) for nr in range(n_rigids) ]
            free_lengths = [ rigid_ind_l[0][0] ]
            for nr in range(n_rigids-1):
                free_lengths += [ rigid_ind_l[nr+1][0] - rigid_ind_l[nr][-1] -1 ]
            free_lengths += [ chain_length-1 - rigid_ind_l[-1][-1] ]
            rigid_dict[key] = {
                                "n_rigids": n_rigids, 
                                "rigid_lengths": rigid_lengths, 
                                "free_lengths": free_lengths,
                                "n_chains": int(mol_dict['N'])
                                }
    return rigid_dict

def reordering_index(syslist):
    n_mols = len(syslist)
    mol_keys = [syslist[mol]['mol'] for mol in range(n_mols)]
    rigid_dict = rigid_dict_from_syslist(syslist)

    reordered_list = []
    n_prev_freeR = 0
    n_prev_rig = np.sum([ rigid_dict[key]['n_rigids']*rigid_dict[key]['n_chains'] for key in mol_keys ])
    n_prev_rig += np.sum([ np.sum(rigid_dict[key]['free_lengths'])*rigid_dict[key]['n_chains'] for key in mol_keys ])
    for mol in range(n_mols):
        key = mol_keys[mol]
        if rigid_dict[key]['n_rigids']==0:
            tmp_length = rigid_dict[key]['free_lengths']
            for ch in range(rigid_dict[key]['n_chains']):
                reordered_list += [n_prev_freeR+i for i in range(tmp_length[0])]
                n_prev_freeR += tmp_length[0]
        else:
            tmp_length_free = rigid_dict[key]['free_lengths']
            tmp_length_rig = rigid_dict[key]['rigid_lengths']
            tmp_reord_list = []
            for ch in range(rigid_dict[key]['n_chains']):
                tmp_reord_list += [n_prev_freeR+i for i in range(rigid_dict[key]['n_rigids']+tmp_length_free[0])]
                n_prev_freeR += rigid_dict[key]['n_rigids']+tmp_length_free[0]
                for nr in range(rigid_dict[key]['n_rigids']):
                    tmp_reord_list += [n_prev_rig+i for i in range(tmp_length_rig[nr])]
                    tmp_reord_list += [n_prev_freeR+i for i in range(tmp_length_free[nr+1])]
                    n_prev_freeR += tmp_length_free[nr+1]
                    n_prev_rig += tmp_length_rig[nr]
            reordered_list += tmp_reord_list

    return reordered_list

    
def create_init_configuration(syslist, aa_param_dict, box_length):
    n_mols = len(syslist)
    n_chains = np.sum([int(syslist[i]['N']) for i in range(n_mols)])
    aa_type = list(aa_param_dict.keys())
    
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
        chain_id, chain_mass, chain_charge, chain_sigma, chain_pos = aa_stats_sequence(mol_dict['pdb'], aa_param_dict)
        chain_length = len(chain_id)
        chain_lengths_list += [chain_length]
        chain_rel_pos = chain_positions_from_pdb(mol_dict['pdb'], relto='com', chain_mass=chain_mass)   # positions relative to c.o.m. 
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
            #s.particles.body += [-1]*chain_length*n_mol_chains
            
            s.bonds.N += len(bond_pairs)
            s.bonds.typeid += [0]*len(bond_pairs)
            s.bonds.group += bond_pairs

            n_prev_res += chain_length*n_mol_chains
            
        ## case with rigid bodies
        else:
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
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
            body_list = [len(s.particles.types)-len(aa_type)+r for r in range(n_rigids)]
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
                I = protein_moment_inertia(rigid_rel_pos, rigid_mass)
                I_diag, E_vec = np.linalg.eig(I)
                moment_inertia_rigid_bodies += [[I_diag[0], I_diag[1], I_diag[2]]]
                # create rigid body object 
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
                    body_list += [-1]*len(free_ind)
                        
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
                body_list += [-1]*len(free_ind)
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
            #s.particles.body += n_mol_chains*body_list
            # bonds
            s.bonds.N += len(bond_pairs)
            s.bonds.typeid += [0]*len(bond_pairs)
            s.bonds.group += bond_pairs
        
            n_prev_res += (n_rigids+length_free_total)*n_mol_chains

        n_prev_mol += n_mol_chains
    
    bond_pairs_tot = s.bonds.group

    ### BUILD RIGID BODIES
    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_snapshot(s)
    rigid.create_bodies(sim.state)
    integrator = hoomd.md.Integrator(dt=0.01, integrate_rotational_dof=True)
    integrator.rigid = rigid
    sim.operations.integrator = integrator
    sim.run(0)
 
    hoomd.write.GSD.write(state=sim.state, filename='2complete_try_start.gsd', mode='wb')

    ### ADD BONDS FREE-RIGID 
    s1 = gsd.hoomd.open(name='2complete_try_start.gsd', mode='r+')[0]
    
    ## indexing
    reordered_ind = reordering_index(syslist)

    ## bonds free rigid
    bonds_free_rigid = []
    n_prev_res = 0
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        if mol_dict['rigid']!='0':
            rigid_ind_l = read_rigid_indexes(mol_dict['rigid'])
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
    with gsd.hoomd.open(name='2complete_try_start.gsd', mode='w') as fout:
        fout.append(s1)
        fout.close()



# ------------------------- MAIN -------------------------------

if __name__=='__main__':
    box_length=50

    stat_file = '../input_stats/stats_module.dat'
    sysfile = 'sys_complete_try.dat'

    aa_param_dict = aa_stats_from_file(stat_file)
    syslist = system_from_file(sysfile)

    try_reord_l = reordering_index(syslist)

    create_init_configuration(syslist, aa_param_dict, box_length)

    '''
        print(s.particles.N)
    print(s.particles.types)
    print(s.particles.typeid)
    print(s.particles.body)
    print(s.bonds.group)
    with gsd.hoomd.open(name='2complete_start.gsd', mode='w') as fout:
        fout.append(s)
        fout.close()
    exit()
    '''