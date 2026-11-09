"""
Residue-residue contact maps between groups of chains (MDAnalysis).

- contact_cutoffs: cutoff of every residue pair from the bead sigmas, one matrix per chain of group 1
- interchain_contact_array: contacts in one frame, summed over chain pairs
- InterchainContacts: contact probability per residue pair per chain pair, averaged over frames
"""
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.distances import distance_array


def contact_cutoffs(sigma_1, sigma_2, factor=1.1):
    """factor * 2^(1/6) * (sigma_i + sigma_j) / 2 for every residue pair.

    sigma_1: (n_chain_1, n_residue_1) bead sigmas of every chain of group 1 (chains may differ, e.g. pSer)
    sigma_2: (n_residue_2,) bead sigmas of one chain of group 2 (all its copies identical)
    Returns (n_chain_1, n_residue_1, n_residue_2), in the units of the sigmas.
    """
    sigma_1, sigma_2 = np.asarray(sigma_1, float), np.asarray(sigma_2, float)
    return factor * 2**(1 / 6) * 0.5 * (sigma_1[:, :, None] + sigma_2[None, None, :])


def interchain_contact_array(positions, n_chain_1, n_chain_2, n_residue_1, n_residue_2, dimensions, cutoff):
    """Contacts (distance <= cutoff) between every residue of the n_chain_1 chains of group 1 and every
    residue of the n_chain_2 chains of group 2, summed over chain pairs: array (n_residue_1, n_residue_2).

    Group 1 is the first n_chain_1 * n_residue_1 particles, group 2 follows it.
    cutoff: scalar, (n_residue_1, n_residue_2) or (n_chain_1, n_residue_1, n_residue_2).
    """
    n_1 = n_chain_1 * n_residue_1
    pos_1 = positions[:n_1]
    contact_ar = np.zeros((n_residue_1, n_residue_2))
    for j in range(n_chain_2):
        pos_2 = positions[n_1 + j * n_residue_2:n_1 + (j + 1) * n_residue_2]
        dist = distance_array(pos_1, pos_2, box=dimensions, backend='OpenMP')
        contact_ar += (dist.reshape(n_chain_1, n_residue_1, n_residue_2) <= cutoff).sum(axis=0)
    return contact_ar


class InterchainContacts(AnalysisBase):
    """interchain_contact_array averaged over the frames of run(start, stop, step).

    After run():
        results.contact_map  contact probability per residue pair per chain pair, (n_residue_1, n_residue_2)
        results.n_frames     frames used
    Usage: InterchainContacts(universe.atoms, n_chain_1, n_chain_2, n_residue_1, n_residue_2, cut_off).run(start, stop, step)
    """

    def __init__(self, atomgroup, n_chain_1, n_chain_2, n_residue_1, n_residue_2, cut_off, verbose=True):
        super().__init__(atomgroup.universe.trajectory, verbose=verbose)
        self.atomgroup = atomgroup
        self.n_chain_1, self.n_residue_1 = n_chain_1, n_residue_1
        self.n_chain_2, self.n_residue_2 = n_chain_2, n_residue_2
        self.cut_off = cut_off

    def _prepare(self):
        self._sum = np.zeros((self.n_residue_1, self.n_residue_2))

    def _single_frame(self):
        self._sum += interchain_contact_array(self.atomgroup.positions, self.n_chain_1, self.n_chain_2,
                                              self.n_residue_1, self.n_residue_2, self._ts.dimensions,
                                              self.cut_off)

    def _conclude(self):
        self.results.n_frames = self.n_frames
        self.results.contact_map = self._sum / self.n_frames / self.n_chain_1 / self.n_chain_2
