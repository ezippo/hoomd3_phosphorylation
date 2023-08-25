import numpy as np
import logging

import hps_phosphorylation.hoomd_util as hu
import hoomd

def metropolis_boltzmann(dU, dmu, kT=2.494338):
    x = np.random.rand()
    if np.log(x) <= -(dU+dmu)/kT:
        return True
    else:
        return False

# ### CUSTOM ACTIONS

class ChangeSerine(hoomd.custom.Action):

    def __init__(self, active_serials, ser_serials, forces, glb_contacts, temp, Dmu, box_size, contact_dist):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._forces = forces
        self._glb_contacts = glb_contacts
        self._temp = temp
        self._Dmu = Dmu
        self._box_size = box_size
        self._contact_dist = contact_dist

    def act(self, timestep):
        snap = self._state.get_snapshot()
        positions = snap.particles.position
        active_pos = positions[self._active_serials]
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials], self._box_size)
        distances = np.max(distances, axis=0)
        min_dist = np.min(distances)

        if min_dist<self._contact_dist:
            ser_index = self._ser_serials[np.argmin(distances)]

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


class ChangeSerineNESS(hoomd.custom.Action):

    def __init__(self, active_serials, ser_serials, forces, glb_contacts, glb_changes, temp, Dmu, box_size, contact_dist):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._forces = forces
        self._glb_contacts = glb_contacts
        self._glb_changes = glb_changes
        self._temp = temp
        self._Dmu = Dmu
        self._box_size = box_size
        self._contact_dist = contact_dist

    def act(self, timestep):
        snap = self._state.get_snapshot()
        positions = snap.particles.position
        active_pos = positions[self._active_serials]
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials], self._box_size)
        distances = np.max(distances, axis=0)
        min_dist = np.min(distances)

        if min_dist<self._contact_dist:
            ser_index = self._ser_serials[np.argmin(distances)]

            if snap.particles.typeid[ser_index]==15:
                U_in = self._forces[0].energy + self._forces[1].energy
                snap.particles.typeid[ser_index] = 20
                self._state.set_snapshot(snap)
                U_fin = self._forces[0].energy + self._forces[1].energy
                logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                if metropolis_boltzmann(U_fin-U_in, self._Dmu, self._temp):
                    logging.info(f"Phosphorylation occured: SER id {ser_index}")
                    self._glb_contacts += [[timestep, ser_index, 1, min_dist, U_fin-U_in]]
		    self._glb_changes += [[timestep, ser_index, 1, min_dist, U_fin-U_in]]
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
		    self._glb_changes += [[timestep, ser_index, -1, min_dist, U_fin-U_in]]
                else:
                    snap.particles.typeid[ser_index] = 20
                    self._state.set_snapshot(snap)
                    logging.info(f'Dephosphorylation SER id {ser_index} not accepted')
                    self._glb_contacts += [[timestep, ser_index, 2, min_dist, U_fin-U_in]]

            else:
                raise Exception(f"Residue {ser_index} is not a serine!")


class ReservoirExchange(hoomd.custom.Action):

    def __init__(self, active_serials, ser_serials, forces, glb_changes, temp, Dmu, box_size, bath_dist):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._forces = forces
        self._temp = temp
        self._Dmu = Dmu
        self._glb_changes = glb_changes
        self._box_size = box_size
        self._bath_dist = bath_dist
        
    def act(self, timestep):
        snap = self._state.get_snapshot()
        positions = snap.particles.position
        active_pos = positions[self._active_serials]
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials], self._box_size)
        distances = np.min(distances, axis=0)
        min_dist = np.min(distances)
        
        if min_dist>self._bath_dist:
            ser_index = self._ser_serials[np.argmin(distances)]

            if snap.particles.typeid[ser_index]==15:
                U_in = self._forces[0].energy + self._forces[1].energy
                snap.particles.typeid[ser_index] = 20
                self._state.set_snapshot(snap)
                U_fin = self._forces[0].energy + self._forces[1].energy
                logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                if metropolis_boltzmann(U_fin-U_in, 0, self._temp):
                    self._glb_changes += [[timestep, ser_index, 10, min_dist, U_fin-U_in]]
                else:
                    snap.particles.typeid[ser_index] = 15
                    self._state.set_snapshot(snap)
                        
            elif snap.particles.typeid[ser_index]==20:
                U_in = self._forces[0].energy + self._forces[1].energy
                snap.particles.typeid[ser_index] = 15
                self._state.set_snapshot(snap)
                U_fin = self._forces[0].energy + self._forces[1].energy
                logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                if metropolis_boltzmann(U_fin-U_in, 0, self._temp):
                    self._glb_changes += [[timestep, ser_index, -10, min_dist, U_fin-U_in]]
                else:
                    snap.particles.typeid[ser_index] = 20
                    self._state.set_snapshot(snap)
                        
            else:
                raise Exception(f"Residue {ser_index} is not a serine!")


class ContactDetector(hoomd.custom.Action):

    def __init__(self, active_serials, ser_serials, glb_contacts, box_size, contact_dist):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._glb_contacts = glb_contacts
        self._box_size = box_size
        self._contact_dist = contact_dist

    def act(self, timestep):
        snap = self._state.get_snapshot()
        positions = snap.particles.position
        active_pos = positions[self._active_serials]
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials], self._box_size)
        distances = np.max(distances, axis=0)
        logging.debug(f"ChangeSerine: distances {distances}")
        min_dist = np.min(distances)
        if min_dist<self._contact_dist:
            ser_index = self._ser_serials[np.argmin(distances)]
            logging.debug(f"ChangeSerine: ser_index {ser_index}")
            self._glb_contacts += [[timestep, ser_index, -2, min_dist, 0.]]
            

class ContactsBackUp(hoomd.custom.Action):

    def __init__(self, glb_contacts, logfile):
        self._glb_contacts = glb_contacts
        self._logfile = logfile

    def act(self, timestep):
        np.savetxt(self._logfile+"_contactsBCKP.txt", self._glb_contacts, fmt='%f')


class ChangesBackUp(hoomd.custom.Action):

    def __init__(self, glb_changes):
        self._glb_changes = glb_changes

    def act(self, timestep):
        np.savetxt(logfile+"_changesBCKP.txt", self._glb_changes, fmt='%f')


def phosphosites_from_syslist(syslist, type_id, chain_lengths_l, n_rigids_l):
    
    reordered_list = hu.reordering_index(syslist)
    n_mols = len(syslist)
    phosphosites = []
    prev_res = 0     

    for mol in range(n_mols):
        mol_dict = syslist[mol]
        n_mol_chains = int(mol_dict['N'])
        end_index = int(n_mol_chains * (n_rigids_l[mol] + chain_lengths_l[mol]))
        
        phospho_sites = mol_dict['phospho_sites']
        
        if phospho_sites == '0':
            tmp_serials = []
        elif phospho_sites.startswith('SER'):
            ser_specific = phospho_sites.rsplit(":")
            
            if len(ser_specific) == 1:
                type_list = [type_id[i] for i in reordered_list[prev_res:prev_res + end_index]]
                tmp_serials = prev_res + np.where(np.isin(type_list, [15, 20]))[0]
            elif len(ser_specific) == 2:
                type_list = [type_id[i] for i in reordered_list[prev_res:prev_res + end_index]]
                start_ser_ind, end_ser_ind = np.array(ser_specific[1].rsplit("-"), dtype=int) - 1
                tmp_mask = np.isin(type_list, [15, 20])
                for nc in range(n_mol_chains):
                    tmp_mask[nc*(n_rigids_l[mol] + chain_lengths_l[mol]):nc*(n_rigids_l[mol] + chain_lengths_l[mol]) + start_ser_ind] = False
                    tmp_mask[nc*(n_rigids_l[mol] + chain_lengths_l[mol]) + end_ser_ind + 1:(nc + 1)*(n_rigids_l[mol] + chain_lengths_l[mol])] = False
                tmp_serials = prev_res + np.where(tmp_mask)[0]
            else:
                raise ValueError(f"phospho-sites are not correctly specified in molecule {mol_dict['mol']}")
        else:
            tmp_list = list(map(int, phospho_sites.rsplit(',')))
            tmp_serials = []
            for nc in range(n_mol_chains):
                tmp_serials += list(np.array(tmp_list) + prev_res + nc * (n_rigids_l[mol] + chain_lengths_l[mol]))
        
        phosphosites += [reordered_list[i] for i in tmp_serials]
        prev_res += end_index
    
    return phosphosites


def activesites_from_syslist(syslist, chain_lengths_l, n_rigids_l):
    
    reordered_list = hu.reordering_index(syslist)
    n_mols = len(syslist)
    activesites = []
    prev_res = 0 
    
    for mol in range(n_mols):
        mol_dict = syslist[mol]
        n_mol_chains = int(mol_dict['N'])
        n_mol_residues = n_rigids_l[mol] + chain_lengths_l[mol]   
        active_sites = mol_dict['active_sites']
        
        if active_sites != '0':
            active_sites_list = list(map(int, active_sites.split(',')))
            
            active_serials_per_chain = [
                [reordered_list[i] for i in list(np.array(active_sites_list) + prev_res + nc * n_mol_residues)]
                for nc in range(n_mol_chains)
            ]
            
            activesites.extend(active_serials_per_chain)
        
        prev_res +=n_mol_chains*n_mol_residues
    
    return activesites
        
        
if __name__=='__main__':
    infile = 'tests/sim0_try/input0.in'
    macro_dict = hu.macros_from_infile(infile)
    aa_param_dict = hu.aa_stats_from_file(macro_dict['stat_file'])
    syslist = hu.system_from_file(macro_dict['sysfile'])
    chain_lengths_l = [40,414,415]
    n_rigids_l = [80, 76, 71, 292]
    ser_serials = phosphosites_from_syslist(syslist, chain_lengths_l, n_rigids_l)
    print(ser_serials)
