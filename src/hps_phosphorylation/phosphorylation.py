import numpy as np
import hps_phosphorylation.hoomd_util as hu


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


class ContactsBackUp(hoomd.custom.Action):

    def __init__(self, glb_contacts):
        self._glb_contacts = glb_contacts

    def act(self, timestep):
        np.savetxt(logfile+"_contactsBCKP.txt", self._glb_contacts, fmt='%f', header="# timestep    SER index    acc    distance     dU  \n# acc= {0->phospho rejected, 1->phospho accepted, 2->dephospho rejected, -1->dephospho accepted} ")
