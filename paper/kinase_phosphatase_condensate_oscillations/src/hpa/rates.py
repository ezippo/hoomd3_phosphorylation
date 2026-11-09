"""
Phosphorylation reactions from the reaction (contacts) files and the dumps.

- back_reaction_mask: immediate back-reactions (same bead and enzyme, reversed one reaction step later)
- count_contacts: (de)phosphorylations per serine from the reaction files
- pSer_per_chain: pSer per chain in time from the dumps

Replicas are given as a number n (files sim1..sim{n}) or as a list of file numbers.
"""
import numpy as np
import gsd.hoomd
from tqdm import tqdm


def _sims(n_sims):
    if isinstance(n_sims, int):
        return list(range(1, n_sims + 1))
    if isinstance(n_sims, list):
        return n_sims
    raise ValueError('n_sims must be int or list of int!')


def back_reaction_mask(data, reaction_dt=200):
    """Rows of a reaction file (columns: time, bead, change, distance, ..., enzyme in column 5) that belong to an
    accepted reaction reversed by the same enzyme on the same bead exactly reaction_dt MD steps later (both rows).
    The file must be sorted in time; consecutive rows are compared."""
    t, bead, change, enz = data[:, 0], data[:, 1], data[:, 2], data[:, 5]
    pair = ((bead[:-1] == bead[1:]) & (enz[:-1] == enz[1:]) & (np.abs(change[:-1]) == 1) &
            (change[:-1] == -change[1:]) & (np.abs(t[1:] - t[:-1]) == reaction_dt))
    mask = np.zeros(len(data), dtype=bool)
    mask[:-1] |= pair
    mask[1:] |= pair
    return mask


def count_contacts(dirpath, file_suffix, ser_l, n_sims, type_of_contact=None, len_prot=154, n_prot=1, start=None,
                   end=None, max_dist=None, nenz=None, reaction_dt=200, remove_back_reactions=True):
    """Number of reactions at every serine of ser_l, summed over the n_prot chains, for every replica.

    Files: dirpath/sim{s}_{file_suffix}, s in 1..n_sims (int) or in n_sims (list).
    type_of_contact: None (all rows), +1 accepted phosphorylations, -1 accepted dephosphorylations,
                     0 rejected phosphorylations, +2 rejected dephosphorylations.
    start, end: keep start < time < end; max_dist: keep distance (column 3) < max_dist; nenz: list of enzyme ids.
    remove_back_reactions: drop the immediate back-reactions (see back_reaction_mask) before counting.

    Returns the mean over the replicas and its standard error, arrays of len(ser_l).
    """
    counts = []
    for s in _sims(n_sims):
        tmp = np.loadtxt(f'{dirpath}/sim{s}_{file_suffix}', ndmin=2)
        if remove_back_reactions:
            tmp = tmp[~back_reaction_mask(tmp, reaction_dt)]
        if type_of_contact is not None:
            tmp = tmp[tmp[:, 2] == type_of_contact]
        if start is not None:
            tmp = tmp[tmp[:, 0] > start]
        if end is not None:
            tmp = tmp[tmp[:, 0] < end]
        if max_dist is not None:
            tmp = tmp[tmp[:, 3] < max_dist]
        if nenz is not None:
            tmp = tmp[np.isin(tmp[:, 5], nenz)]

        residue = tmp[:, 1].astype(int) % len_prot          # residue index of every reaction in its chain
        in_chains = tmp[:, 1] < len_prot * n_prot
        counts.append([np.sum(in_chains & (residue == i)) for i in ser_l])

    counts = np.array(counts, dtype=float)
    return counts.mean(axis=0), counts.std(axis=0, ddof=1) / np.sqrt(len(counts))


def pSer_per_chain(dirpath, file_suffix, n_sims, times, len_prot=154, n_prot=2, phospho_typeid=20, error='chains'):
    """Mean number of pSer per chain at the frames `times` of every replica (files dirpath/sim{s}_{file_suffix}).

    error: 'chains'   standard error over all (replica x chain) values
           'replicas' standard error of the replica means (chains of one replica are not independent)
    Returns mean and standard error, arrays of len(times).
    """
    sims = _sims(n_sims)
    values = np.zeros((len(sims), n_prot, len(times)))
    for ns, s in enumerate(sims):
        with gsd.hoomd.open(f'{dirpath}/sim{s}_{file_suffix}', 'r') as traj:
            for i, tt in enumerate(tqdm(times)):
                type_ids = traj[int(tt)].particles.typeid[:len_prot * n_prot].reshape(n_prot, len_prot)
                values[ns, :, i] = np.sum(type_ids == phospho_typeid, axis=1)

    if error == 'replicas':
        per_replica = values.mean(axis=1)
        return per_replica.mean(axis=0), per_replica.std(axis=0, ddof=1) / np.sqrt(len(sims))
    flat = values.reshape(-1, len(times))
    return flat.mean(axis=0), flat.std(axis=0, ddof=1) / np.sqrt(len(flat))
