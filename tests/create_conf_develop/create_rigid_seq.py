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
box_length=50

stat_file = '../input_stats/stats_module.dat'
filein_tdp43 = '../input_stats/CA_TDP-43_261truncated.pdb'
sysfile = 'sys_ck1d_try.dat'

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
        print((chain_pos-chain_com_pos).shape)
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
    
# ------------------------- MAIN -------------------------------

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
        aa_sigma.append(aa_param_dict[k][2])
        aa_lambda.append(aa_param_dict[k][3])

    syslist = system_from_file(sysfile)
    n_mols = len(syslist)
    n_chains = np.sum([int(syslist[i]['N']) for i in range(n_mols)])
    aa_type = list(aa_param_dict.keys())
    
    K = math.ceil(n_chains**(1/3))
    spacing = box_length/K
    x = np.linspace(-box_length/2, box_length/2, K, endpoint=False)
    positions = list(itertools.product(x, repeat=3))
    positions = np.array(positions) + [spacing/2, spacing/2, spacing/2]
    np.random.shuffle(positions)
    positions = positions[:n_chains, :]
    
    mol_dict = syslist[0]
    chain_id, chain_mass, chain_charge, chain_sigma, chain_pos = aa_stats_sequence(mol_dict['pdb'], aa_param_dict)
    chain_length = len(chain_id)
    chain_rel_pos = chain_positions_from_pdb(mol_dict['pdb'], relto='com', chain_mass=chain_mass)   # positions relative to c.o.m. 
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
        moment_inertia_rigid_bodies += [[I_diag[0], I_diag[1], I_diag[2]]]
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

    ## set first snapshot
    s=gsd.hoomd.Frame()
    s.configuration.dimensions = 3
    s.configuration.box = [box_length,box_length,box_length,0,0,0] 
    s.configuration.step = 0
    # particles
    s.particles.N = n_rigids + np.sum(length_free_bodies)
    s.particles.types = aa_type + types_rigid_bodies
    s.particles.typeid = typeid_rigid_bodies + typeid_free_bodies
    s.particles.mass = mass_rigid_bodies + mass_free_bodies
    s.particles.charge = [0]*n_rigids + charge_free_bodies
    s.particles.position = position_rigid_bodies + position_free_bodies
    s.particles.moment_inertia = moment_inertia_rigid_bodies + [[0,0,0]]*np.sum(length_free_bodies)
    s.particles.orientation = [(1, 0, 0, 0)]*(n_rigids+np.sum(length_free_bodies))
    s.particles.body = [r for r in range(n_rigids)] + [-1]*np.sum(length_free_bodies)
    # bonds
    s.bonds.N = 0
    s.bonds.types = ['AA_bond']
    s.bonds.typeid = []
    s.bonds.group = []
    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_snapshot(s)
    rigid.create_bodies(sim.state)

    integrator = hoomd.md.Integrator(dt=0.01, integrate_rotational_dof=True)
    integrator.rigid = rigid
    sim.operations.integrator = integrator
    sim.run(0)
    hoomd.write.GSD.write(state=sim.state, filename='ck1d_try_start.gsd', mode='wb')

    ## bonds
    bond_pairs = []
    n_prev_res = 0
    for length in length_free_bodies:
        bond_pairs += [ [n_rigids+n_prev_res+i, n_rigids+n_prev_res+i+1 ] for i in range(length-1)]
        n_prev_res += int(length)
    # add bonds free-rigid
    for b in range(len(bonds_free_rigid)):
        if bonds_free_rigid[b][1]<0:
            r = -bonds_free_rigid[b][1] 
            length = np.sum([ len(rigid.body['R'+str(r_ind)]['charges']) for r_ind in range(1,r+1) ])
            bonds_free_rigid[b][1] = n_rigids + np.sum(length_free_bodies) + length -1
        else:
            r = bonds_free_rigid[b][1] 
            length = np.sum([ len(rigid.body['R'+str(r_ind)]['charges']) for r_ind in range(1,r) ])
            bonds_free_rigid[b][1] = n_rigids + np.sum(length_free_bodies) + int(length) 
    bond_pairs += bonds_free_rigid

    s1 = gsd.hoomd.open(name='ck1d_try_start.gsd', mode='r+')[0]
    #s = gsd.hoomd.open('ck1d_try_start.gsd', 'rb')[0]
    s1.bonds.N = len(bond_pairs) 
    s1.bonds.typeid = [0]*len(bond_pairs)
    s1.bonds.group = bond_pairs
    
    print(s1.particles.body)
    with gsd.hoomd.open(name='ck1d_try_start.gsd', mode='w') as fout:
        fout.append(s1)
        fout.close()
 
