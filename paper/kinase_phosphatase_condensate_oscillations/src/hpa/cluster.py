"""
Size of the condensate (largest cluster of substrate chains) and pSer content in time.

- from the dumps: condensate_size_with_freud (one frame), chains_in_condensate (trajectories)
- from the analysis files sim{ss}_size_cond_*.dat: read_chains_in_condensate
"""
import numpy as np
import gsd.hoomd
import freud
from tqdm import tqdm


def condensate_size_with_freud(frame, n_particles=30800, cutoff=1.0, n_clusters=1):
    """Number of substrate beads (the first n_particles) in the largest cluster(s) of a GSD frame."""
    box_f = freud.box.Box.from_box(frame.configuration.box)
    freudcl = freud.cluster.Cluster()
    freudcl.compute((box_f, frame.particles.position), neighbors={'r_max': cutoff})
    _, counts = np.unique(freudcl.cluster_idx[:n_particles], return_counts=True)
    largest = np.sort(counts)[::-1][:n_clusters]
    return largest[0] if n_clusters == 1 else largest


def chains_in_condensate(dirpath, file_suffix, n_sims, times, n_particles=30800, cutoff=1.0, n_clusters=1,
                         chain_length=154, phospho_typeid=20):
    """Chains in the largest cluster(s) and number of pSer at the frames `times`, averaged over replicas.

    Files: dirpath + f'sim{s}_' + file_suffix, with s in range(1, n_sims + 1) (int), in n_sims (list),
    or '' (None).
    Returns n_chains (len(times), n_clusters) and n_pSer (len(times),).
    """
    if isinstance(n_sims, int):
        sims_list = list(range(1, n_sims + 1))
    elif isinstance(n_sims, list):
        sims_list = n_sims
    elif n_sims is None:
        sims_list = ['']
    else:
        raise ValueError('n_sims must be int or list of int!')
    nsims = len(sims_list)

    n_chains_arr = np.zeros((len(times), n_clusters))
    n_phospho_arr = np.zeros(len(times))

    for s in sims_list:
        tmp_nc = np.zeros((len(times), n_clusters))
        tmp_np = np.zeros(len(times))
        with gsd.hoomd.open(dirpath + f'sim{s}_' + file_suffix, 'rb') as input_gsd:
            print(len(input_gsd), nsims)
            for i, tt in enumerate(tqdm(times)):
                frame = input_gsd[int(tt)]
                sizes = condensate_size_with_freud(frame, n_particles, cutoff, n_clusters)
                if n_clusters == 1:
                    tmp_nc[i, 0] = sizes / chain_length
                else:
                    tmp_nc[i, :len(sizes)] = sizes / chain_length
                tmp_np[i] = np.sum(frame.particles.typeid == phospho_typeid)
        n_chains_arr += tmp_nc
        n_phospho_arr += tmp_np

    return n_chains_arr / nsims, n_phospho_arr / nsims


def read_chains_in_condensate(dirpath, n_chains, n_chains_enz1, n_chains_enz2, sim_type, n_sim, fr='_fr01',
                              therm=0, nB=False):
    """Time, chains in the condensate and total pSer from a size file of the droplet simulations.

    File: dirpath/sim_{sim_type}_{n_chains}-tdp43-lcd_{n_chains_enz1}-ck1d-open_{n_chains_enz2}-pp3{fr}/
          sim{n_sim}_size_cond_..._fr01.dat (one row every 4e6 MD steps = 0.04 us)
    therm : number of rows (frames) to skip
    nB : also return the 4th column (number of B chains), if present
    """
    name = f'{sim_type}_{n_chains}-tdp43-lcd_{n_chains_enz1}-ck1d-open_{n_chains_enz2}-pp3{fr}'
    data = np.loadtxt(dirpath + f'sim_{name}/sim{n_sim}_size_cond_{name}.dat')
    if nB:
        return data[therm:, 0], data[therm:, 1], data[therm:, 2], data[therm:, 3]
    return data[therm:, 0], data[therm:, 1], data[therm:, 2]
