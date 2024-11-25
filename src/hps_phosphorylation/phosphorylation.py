import numpy as np
import logging

import hps_phosphorylation.hoomd_util as hu
import hoomd

def metropolis_boltzmann(dU, dmu, kT=2.494338):
    """
    Metropolis-Boltzmann acceptance criterion to decide whether to accept or reject a state change.
    
    Args:
        dU (float): Energy difference between the current and proposed state.
        dmu (float): Chemical potential difference between the current and proposed state.
        kT (float, optional): Thermal energy in kJ/mol, defaults to 2.494338 (=300K).
    
    Returns:
        bool: True if the state change is accepted, False otherwise.
    """
    random_number = np.random.rand()
    if np.log(random_number) <= -(dU+dmu)/kT:
        return True
    else:
        return False



# ### CUSTOM ACTIONS

class ChangeSerine(hoomd.custom.Action):
    """
    Custom Action to handle the phosphorylation and dephosphorylation of serines
    during molecular dynamics simulations.
    Compute distances between serines (or phospho-serine) in ser_serials and ezyme active site residues in active_serials. 
    Get the closest serine. 
    If all the distances with the active site residues are below the contact threshold contact_dist, attempt the reaction with Metropolis acceptance.
    Modify simulation state if needed and save attempt result in list glb_contacts.

    Args:
        active_serials (list): List of enzyme active site serial numbers.
        ser_serials (list): List of serine serial numbers.
        forces (list): List of pair potential objects to compute energy differences.
        glb_contacts (list): Global list to record contact events.
        temp (float): Temperature of the system (in energy units) for the Metropolis-Boltzmann acceptance.
        Dmu (float): Chemical potential difference for phosphorylation/dephosphorylation Metropolis step.
        box_size (tuple): Size of the simulation box (x, y, z dimensions).
        contact_dist (float): Distance threshold for a contact between the enzyme active site residues and a serine.
        enzyme_ind (int): Index of the enzyme for tracking in case of multiple enzymes.
        glb_changes (list, optional): Global list to record type change events (Ser to pSer or opposite), necessary only in simulation mode 'ness'. Default None.
        id_Ser_types (list, optional): List of IDs number associated with Ser in free chain and rigid body. Default [15] (no rigid body).
        id_pSer_types (list, optional): List of IDs number associated with pSer in free chain and rigid body. Default [20] (no rigid body).

    """
    def __init__(self, active_serials, ser_serials, forces, glb_contacts, temp, Dmu, box_size, contact_dist, enzyme_ind, glb_changes=None, id_Ser_types=[15], id_pSer_types=[20]):
        # Initialize all instance variables
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._forces = forces
        self._glb_contacts = glb_contacts
        self._temp = temp
        self._Dmu = Dmu
        self._box_size = box_size
        self._contact_dist = contact_dist
        self._enzyme_ind = enzyme_ind
        self._glb_changes = glb_changes
        self._id_Ser_types = id_Ser_types
        self._id_pSer_types = id_pSer_types

    def act(self, timestep):
        """
        Executes the phosphorylation or dephosphorylation based on the enzyme-serine proximity and energy.

        Args:
            timestep (int): The current timestep of the simulation, standard act definition (see HOOMD-blue v3 docmentation).

        Raises:
            Exception: If the residue is not Ser or pSer (typeid other than 15 or 20).
        """
        snap = self._state.get_snapshot()     # Get the simulation snapshot
        positions = snap.particles.position      # Get the positions of particles
        active_pos = positions[self._active_serials]     # enzyme active site positions

        # Compute distance
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials],  self._box_size[0], self._box_size[1], self._box_size[2])
        distances = np.max(distances, axis=0)
        min_dist = np.min(distances)   # get distance of the closest serine

        # If the all distances between closest serine and active site residues are below the contact threshold, attempt modification
        if min_dist<self._contact_dist:
            ser_index = self._ser_serials[np.argmin(distances)]   # get closest serine

            for idser in range( len(self._id_Ser_types) ):      # id_Ser_types can contain only SER id, or also SER_r in case of rigid bodies  
                # if closest residue of ser_serials is a Ser, try phosphorylation
                if snap.particles.typeid[ser_index] == self._id_Ser_types[idser]:
                    U_in = self._forces[0].energy + self._forces[1].energy
                    snap.particles.typeid[ser_index] = self._id_pSer_types[idser]
                    self._state.set_snapshot(snap)
                    U_fin = self._forces[0].energy + self._forces[1].energy
                    logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")

                    # Apply the Metropolis criterion for phosphorylation
                    if metropolis_boltzmann(U_fin-U_in, self._Dmu, self._temp):
                        logging.info(f"Phosphorylation occured: SER id {ser_index}")
                        self._glb_contacts += [[timestep, ser_index, 1, min_dist, U_fin-U_in, self._enzyme_ind]]
                        if self._glb_changes is not None:
                            self._glb_changes += [[timestep, ser_index, 1, min_dist, U_fin-U_in, self._enzyme_ind]]

                    else:
                        # Revert if the Metropolis test fails
                        snap.particles.typeid[ser_index] = self._id_Ser_types[idser]
                        self._state.set_snapshot(snap)
                        logging.info(f'Phosphorylation SER id {ser_index} not accepted')
                        self._glb_contacts += [[timestep, ser_index, 0, min_dist, U_fin-U_in, self._enzyme_ind]]
                        
                # if closest residue of ser_serials is a pSer, try de-phosphorylation
                elif snap.particles.typeid[ser_index]==self._id_pSer_types[idser]:
                    U_in = self._forces[0].energy + self._forces[1].energy
                    snap.particles.typeid[ser_index] = self._id_Ser_types[idser]
                    self._state.set_snapshot(snap)
                    U_fin = self._forces[0].energy + self._forces[1].energy
                    logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                    if metropolis_boltzmann(U_fin-U_in, -self._Dmu, self._temp):
                        logging.info(f"Dephosphorylation occured: SER id {ser_index}")
                        self._glb_contacts += [[timestep, ser_index, -1, min_dist, U_fin-U_in, self._enzyme_ind]]
                        if self._glb_changes is not None:
                        self._glb_changes += [[timestep, ser_index, -1, min_dist, U_fin-U_in, self._enzyme_ind]]

                    else:
                        snap.particles.typeid[ser_index] = self._id_pSer_types[idser]
                        self._state.set_snapshot(snap)
                        logging.info(f'Dephosphorylation SER id {ser_index} not accepted')
                        self._glb_contacts += [[timestep, ser_index, 2, min_dist, U_fin-U_in, self._enzyme_ind]]

                else:
                    raise Exception(f"Residue {ser_index} is not a serine!")



class ReservoirExchange(hoomd.custom.Action):
    """
    Action for performing reservoir exchange with serine residues in a simulation.

    This class handles the exchange of serine residues between two states (e.g., phosphorylated and dephosphorylated)
    based on their distance from active sites. It uses a Metropolis criterion to decide whether to accept or reject
    the state change based on energy differences.

    Args:
        active_serials (list): List of enzyme active site serial numbers.
        ser_serials (list): List of serine serial numbers.
        forces (list): List of pair potential objects to compute energy differences.
        glb_contacts (list): Global list to record contact events.
        temp (float): Temperature of the system (in energy units) for the Metropolis-Boltzmann acceptance.
        Dmu (float): Chemical potential difference for phosphorylation/dephosphorylation Metropolis step.
        box_size (tuple): Size of the simulation box (x, y, z dimensions).
        bath_dist (float): Minimum distance threshold for reservoir exchange.
        id_Ser_types (list, optional): List of IDs number associated with Ser in free chain and rigid body. Default [15] (no rigid body).
        id_pSer_types (list, optional): List of IDs number associated with pSer in free chain and rigid body. Default [20] (no rigid body).

    """
    def __init__(self, active_serials, ser_serials, forces, glb_changes, temp, Dmu, box_size, bath_dist, id_Ser_types=[15], id_pSer_types=[20]):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._forces = forces
        self._temp = temp
        self._Dmu = Dmu
        self._glb_changes = glb_changes
        self._box_size = box_size
        self._bath_dist = bath_dist
        self._id_Ser_types = id_Ser_types
        self._id_pSer_types = id_pSer_types
        
    def act(self, timestep):
        """
        Executes the reservoir exchange action at a given timestep.

        Args:
            timestep (int): The current timestep of the simulation, standard act definition (see HOOMD-blue v3 docmentation).
        
        Raises:
            Exception: If the residue is not Ser or pSer (typeid other than 15 or 20).
        """
        snap = self._state.get_snapshot()    # get simulation state
        positions = snap.particles.position    
        active_pos = positions[self._active_serials]     # get active site residues positions
        # compute distance
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials],  self._box_size[0], self._box_size[1], self._box_size[2])
        distances = np.min(distances, axis=0)
        min_dist = np.max(distances)   # get the distance of the farthest Ser from the active site
        
        # if all distances from the farthest Ser to the residues of the active site are larger than bath_dist, attemp to switch particle type
        if min_dist>self._bath_dist:
            ser_index = self._ser_serials[np.argmin(distances)]

            for idser in range( len(self._id_Ser_types) ):      # id_Ser_types can contain only SER id, or also SER_r in case of rigid bodies  
                # if the farthest particle is Ser, try to switch in pSer with Metropolis
                if snap.particles.typeid[ser_index] == self._id_Ser_types[idser]:
                    U_in = self._forces[0].energy + self._forces[1].energy
                    snap.particles.typeid[ser_index] = self._id_pSer_types[idser]
                    self._state.set_snapshot(snap)
                    U_fin = self._forces[0].energy + self._forces[1].energy
                    logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                    if metropolis_boltzmann(U_fin-U_in, 0, self._temp):
                        self._glb_changes += [[timestep, ser_index, 10, min_dist, U_fin-U_in, -1]]
                        logging.debug(f"Reservoir exchange Ser -> pSer: SER id {ser_index}")
                    else:
                        snap.particles.typeid[ser_index] = self._id_Ser_types[idser]
                        self._state.set_snapshot(snap)
                        logging.debug(f"Rejected reservoir exchange Ser -> pSer: SER id {ser_index}")
                            
                # if the farthest particle is pSer, try to switch in Ser with Metropolis
                elif snap.particles.typeid[ser_index] == self._id_pSer_types[idser]:
                    U_in = self._forces[0].energy + self._forces[1].energy
                    snap.particles.typeid[ser_index] = self._id_Ser_types[idser]
                    self._state.set_snapshot(snap)
                    U_fin = self._forces[0].energy + self._forces[1].energy
                    logging.debug(f"U_fin = {U_fin}, U_in = {U_in}")
                    if metropolis_boltzmann(U_fin-U_in, 0, self._temp):
                        self._glb_changes += [[timestep, ser_index, -10, min_dist, U_fin-U_in, -1]]
                        logging.debug(f"Reservoir exchange pSer -> Ser: SEP id {ser_index}")
                    else:
                        snap.particles.typeid[ser_index] = self._id_pSer_types[idser]
                        self._state.set_snapshot(snap)
                        logging.debug(f"Rejected reservoir exchange pSer -> Ser: SEP id {ser_index}")
                            
                else:
                    raise Exception(f"Residue {ser_index} is not a serine!")


class ContactDetector(hoomd.custom.Action):
    """
    Action for detecting contacts between active sites and serine residues in a simulation.

    This class detects when any active site comes within a specified distance of serine residues.
    It records these contact events along with relevant details.

    Args:
        active_serials (list of int): Indices of the active sites.
        ser_serials (list of int): Indices of the serine residues.
        glb_contacts (list of list): List to record contact events.
        box_size (list of float): Size of the simulation box.
        contact_dist (float): Distance threshold for detecting contacts.
        enzyme_ind (int): Index of the enzyme.
        displ_as_pos (ndarray, optional): Array with list of displacement vectors for each active site residue. Default None, no displacement.
        reference_vector (ndarray, optional): Array with reference vector to compute the rotation of the rigid body with the active site. Needed in case of displacement (displ_as_pos not None). Default None.
"""
    def __init__(self, active_serials, ser_serials, glb_contacts, box_size, contact_dist, enzyme_ind):
        self._active_serials = active_serials
        self._ser_serials = ser_serials
        self._glb_contacts = glb_contacts
        self._box_size = box_size
        self._contact_dist = contact_dist
        self._enzyme_ind = enzyme_ind
        
    def act(self, timestep):
        """
        Executes the contact detection action at a given timestep.

        Args:
            timestep (int): The current timestep of the simulation, standard act definition (see HOOMD-blue v3 docmentation).
        """
        snap = self._state.get_snapshot()     # get simulation state
        positions = snap.particles.position  
        active_pos = positions[self._active_serials]    # get active site positions
            
        # compute distances
        distances = hu.compute_distances_pbc(active_pos, positions[self._ser_serials], self._box_size[0], self._box_size[1], self._box_size[2])
        distances = np.max(distances, axis=0)
        logging.debug(f"ChangeSerine: distances {distances}")
        min_dist = np.min(distances)    # get distance of closest particle in ser_serials to the active site

        # If the all distances between closest serine and active site residues are below the contact threshold, save it as a contact
        if min_dist<self._contact_dist:
            ser_index = self._ser_serials[np.argmin(distances)]
            logging.debug(f"ChangeSerine: ser_index {ser_index}")
            self._glb_contacts += [[timestep, ser_index, -2, min_dist, 0., self._enzyme_ind]]
            

class ContactsBackUp(hoomd.custom.Action):
    """
    Action for backing up contact records to a file.

    This class saves the global contact records to a specified file at each timestep, defined with the hoomd trigger.

    Args:
        glb_contacts (list of list): List containing global contact records.
        logfile (str): Base name of the file where contact records will be saved.
    """
    def __init__(self, glb_contacts, logfile):
        self._glb_contacts = glb_contacts
        self._logfile = logfile

    def act(self, timestep):
        np.savetxt(self._logfile+"_contactsBCKP.txt", self._glb_contacts, fmt='%f')


class ChangesBackUp(hoomd.custom.Action):
    """
    Action for backing up change records to a file.

    This class saves the global change records to a specified file at each timestep.

    Args:
        glb_changes (list of list): List containing global change records.
        logfile (str): Base name of the file where change records will be saved.
    """
    def __init__(self, glb_changes, logfile):
        self._glb_changes = glb_changes
        self._logfile = logfile

    def act(self, timestep):
        np.savetxt(self._logfile+"_changesBCKP.txt", self._glb_changes, fmt='%f')


def phosphosites_from_syslist(syslist, type_id, chain_lengths_l, n_rigids_l, id_ser_types=[15,20]):
    """
    Extracts phosphosite serials from a system list.

    This function processes a system list to determine the indices of phosphosites
    in each molecule based on the type IDs and specified phospho-site information.

    Args:
        syslist (list of dict): List of dictionaries, where each dictionary represents a molecule.
        type_id (list of int): List of type IDs for particles.
        chain_lengths_l (list of int): List of chain lengths for each molecule.
        n_rigids_l (list of int): List of rigid body indices for each molecule.
        id_ser_types (list of int): List of IDs number associated with Ser and pSer in free chain and rigid body. Default [15,20] (no rigid body).

    Returns:
        list of int: List of reordered indices corresponding to phosphosites.
    """
    # Reorder the system list indices for consistency
    reordered_list = hu.reordering_index(syslist)
    n_mols = len(syslist)
    phosphosites = []
    prev_res = 0     

    for mol in range(n_mols):
        mol_dict = syslist[mol]
        n_mol_chains = int(mol_dict['N'])
        end_index = int(n_mol_chains * (n_rigids_l[mol] + chain_lengths_l[mol]))   # last index for molecules of type mol_dict['mol']
        
        # phosphosites specification: '0'=no phosphosites, 'SER'=all Ser/pSer, 'SER:start-end'=Ser/pSer from start index to end index, {x1,x2,x3,...}=residue index x1,x2,x3,...
        phospho_sites = mol_dict['phospho_sites']
        
        if phospho_sites == '0':
            # No phospho-sites specified
            tmp_serials = []

        elif phospho_sites.startswith('SER'):
            ser_specific = phospho_sites.rsplit(":")
            # mask for Ser/pSer residues in type_id of molecules mol_dict['mol']
            tmp_mask = np.isin([type_id[i] for i in reordered_list[prev_res:prev_res + end_index]], id_ser_types)  

            if len(ser_specific) == 2:
                # extract Ser/pSer only in a specific range ('SER:start-end')
                start_ser_ind, end_ser_ind = np.array(ser_specific[1].rsplit("-"), dtype=int) - 1
                # loop over chains of same species
                for nc in range(n_mol_chains):
                    # turn to False the elements of mask outside the range 'start-end'
                    chain_start = nc * (n_rigids_l[mol] + chain_lengths_l[mol])
                    chain_end = chain_start + (n_rigids_l[mol] + chain_lengths_l[mol])
                    tmp_mask[chain_start:chain_start + n_rigids_l[mol] + start_ser_ind] = False
                    tmp_mask[chain_start + n_rigids_l[mol] + end_ser_ind + 1:chain_end] = False
            elif len(ser_specific) != 1:
                raise ValueError(f"phospho-sites are not correctly specified in molecule {mol_dict['mol']}")

            tmp_serials = prev_res + np.where(tmp_mask)[0]
        
        else:
            # extract phosphosites from specific indices
            tmp_list = list(map(int, phospho_sites.rsplit(',')))
            tmp_serials = []
            for nc in range(n_mol_chains):
                tmp_serials += list(np.array(tmp_list)-1 + n_rigids_l[mol] + prev_res + nc * (n_rigids_l[mol] + chain_lengths_l[mol]))
        
        phosphosites += [reordered_list[i] for i in tmp_serials]
        prev_res += end_index
    
    return phosphosites


def activesites_from_syslist(syslist, chain_lengths_l, n_rigids_l):
    """
    Extracts active site indices from a system list.

    Processes the system list to determine the indices of active sites for each molecule.

    Args:
        syslist (list of dict): List of dictionaries, where each dictionary represents a molecule.
        chain_lengths_l (list of int): List of chain lengths for each molecule.
        n_rigids_l (list of int): List of rigid body counts for each molecule.

    Returns:
        list of int: List of reordered indices corresponding to active sites.
    """
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
                [reordered_list[i] for i in list(np.array(active_sites_list)-1 + n_rigids_l[mol] + prev_res + nc * n_mol_residues)]
               for nc in range(n_mol_chains)
            ]
            
            activesites.extend(active_serials_per_chain)
        
        prev_res +=n_mol_chains*n_mol_residues
        
    return activesites
        
        
if __name__=='__main__':
    import gsd
    
    infile = 'examples/sim_try/input_try.in'
    macro_dict = hu.macros_from_infile(infile)
    aa_param_dict = hu.aa_stats_from_file(macro_dict['stat_file'])
    syslist = hu.system_from_file(macro_dict['sysfile'])
    snap = gsd.hoomd.open(macro_dict['file_start'])[0]
    type_id = snap.particles.typeid
    chain_lengths_l = [154,292,415]
    n_rigids_l = [1,2,2]
    active_serials = activesites_from_syslist(syslist, chain_lengths_l, n_rigids_l)
    ser_serials = phosphosites_from_syslist(syslist, type_id, chain_lengths_l, n_rigids_l)
    print(ser_serials)
    print(type_id[ser_serials])
    print(active_serials)
    print(type_id[active_serials])
