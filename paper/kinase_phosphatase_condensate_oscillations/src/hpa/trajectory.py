"""
Trajectory processing under periodic boundary conditions.

- centre of mass and distances under PBC: periodic_mass_weighted_com(_1d), compute_distances_pbc
- centring of GSD trajectories on the condensate: center_trajectory (droplets), center_trajectory_z (slabs)
- distance time series between particle groups: create_distance_file
"""
import copy

import numpy as np
import gsd.hoomd
import freud
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PBC helpers
# ---------------------------------------------------------------------------

def periodic_mass_weighted_com(pos, masses, box_lengths):
    """Mass-weighted COM with circular statistics (PBC-safe), for boxes centred at the origin [-L/2, L/2]."""
    com = np.zeros(3)

    for dim in range(3):
        L = box_lengths[dim]
        theta = 2 * np.pi * pos[:, dim] / L

        sin_sum = np.sum(masses * np.sin(theta))
        cos_sum = np.sum(masses * np.cos(theta))

        angle = np.arctan2(sin_sum, cos_sum)
        com[dim] = L * angle / (2 * np.pi)

    return com


def periodic_mass_weighted_com_1d(pos_1d, masses, L):
    """Mass-weighted periodic COM along a single axis [-L/2, L/2]."""
    theta = 2 * np.pi * pos_1d / L
    s = np.sum(masses * np.sin(theta))
    c = np.sum(masses * np.cos(theta))
    return L * np.arctan2(s, c) / (2 * np.pi)


def wrap_positions(positions, box):
    """Wrap positions (N x 3) into the box [-L/2, L/2)."""
    box_lengths = np.array([box[0], box[1], box[2]])
    return positions - np.floor(positions / box_lengths + 0.5) * box_lengths


def wrap_positions_z(pos, Lz):
    """Wrap the z coordinate into [-Lz/2, Lz/2]."""
    pos[:, 2] -= np.round(pos[:, 2] / Lz) * Lz
    return pos


def largest_cluster_indices(pos, box, cutoff):
    """Indices of the particles in the largest cluster (freud, PBC-aware)."""
    fbox = freud.box.Box.from_box(box)
    cluster = freud.cluster.Cluster()
    cluster.compute((fbox, pos), neighbors={'r_max': cutoff})

    ids = cluster.cluster_idx
    unique, counts = np.unique(ids, return_counts=True)
    largest_id = unique[np.argmax(counts)]

    return np.where(ids == largest_id)[0]


def compute_distances_pbc(p1, p2, x_side, y_side, z_side):
    """Minimum-image distances (N1 x N2) between the particles of p1 (N1 x 3) and p2 (N2 x 3)."""
    dx = np.abs(p1[:, 0, None] - p2[:, 0])
    dy = np.abs(p1[:, 1, None] - p2[:, 1])
    dz = np.abs(p1[:, 2, None] - p2[:, 2])

    dx = np.minimum(dx, x_side - dx)
    dy = np.minimum(dy, y_side - dy)
    dz = np.minimum(dz, z_side - dz)

    return np.sqrt(dx**2 + dy**2 + dz**2)


# ---------------------------------------------------------------------------
# Centring of trajectories
# ---------------------------------------------------------------------------

def center_trajectory(input_gsd, output_gsd, group=None, cluster=False, cutoff=2.5):
    """Write a copy of the trajectory with the mass-weighted periodic COM at the origin.

    group : particle indices used for the COM (all particles if None)
    cluster : if True, use only the largest cluster of `group` (cluster `cutoff` in nm)
    """
    traj_in = gsd.hoomd.open(input_gsd, 'rb')
    traj_out = gsd.hoomd.open(output_gsd, 'wb')

    for frame in tqdm(traj_in, desc="Centering trajectory"):
        new_frame = copy.deepcopy(frame)

        pos = frame.particles.position.copy()
        masses = frame.particles.mass.copy()
        box = frame.configuration.box
        group_ = np.arange(frame.particles.N) if group is None else np.array(group)

        if cluster:
            cluster_global_idx = group_[largest_cluster_indices(pos[group_], box, cutoff)]
            com = periodic_mass_weighted_com(pos[cluster_global_idx], masses[cluster_global_idx], box[:3])
        else:
            com = periodic_mass_weighted_com(pos[group_], masses[group_], box[:3])

        pos -= com                       # in place: positions stay float32
        new_frame.particles.position = wrap_positions(pos, box)
        traj_out.append(new_frame)

    traj_out.close()


def center_trajectory_z(input_gsd, output_gsd, group=None, cluster=False, cutoff=2.5, therm=0):
    """Like center_trajectory, along z only (slab simulations); the first `therm` frames are skipped."""
    traj_in = gsd.hoomd.open(input_gsd, 'rb')[therm:]
    traj_out = gsd.hoomd.open(output_gsd, 'wb')

    for frame in tqdm(traj_in, desc="Centering trajectory along z"):
        new_frame = copy.deepcopy(frame)

        pos = frame.particles.position.copy()
        masses = frame.particles.mass.copy()
        box = frame.configuration.box
        Lz = box[2]
        group_ = np.arange(frame.particles.N) if group is None else np.array(group)

        if cluster:
            cluster_global_idx = group_[largest_cluster_indices(pos[group_], box, cutoff)]
            com_z = periodic_mass_weighted_com_1d(pos[cluster_global_idx, 2], masses[cluster_global_idx], Lz)
        else:
            com_z = periodic_mass_weighted_com_1d(pos[group_, 2], masses[group_], Lz)

        pos[:, 2] -= com_z
        new_frame.particles.position = wrap_positions_z(pos, Lz)
        new_frame.particles.image[:] = 0

        traj_out.append(new_frame)

    traj_out.close()


# ---------------------------------------------------------------------------
# Distances between groups
# ---------------------------------------------------------------------------

def create_distance_file(filename, id1, id2, mode='minimum', therm=0, max_time=None, save=None):
    """Distances between the atom groups id1 and id2 in every frame (MDAnalysis, PBC).

    mode : 'minimum' -> minimum distance (n_frames,)
           'mean1'   -> distance of every atom of id2 averaged over id1 (n_frames, len(id2))
           None      -> full distance matrices (n_frames, len(id1), len(id2))
    """
    u = mda.Universe(filename)
    ag1 = u.atoms[id1]
    ag2 = u.atoms[id2]
    traj = u.trajectory

    start = therm
    end = len(traj) if max_time is None else max_time
    n_steps = end - start

    if mode == 'minimum':
        dist = np.empty(n_steps, dtype=np.float32)
    elif mode == 'mean1':
        dist = np.empty((n_steps, len(ag2)), dtype=np.float32)
    elif mode is None:
        dist = np.empty((n_steps, len(ag1), len(ag2)), dtype=np.float32)
    else:
        raise ValueError("mode must be 'minimum', 'mean1' or None")

    for i, ts in enumerate(tqdm(traj[start:end], total=n_steps)):
        d = distance_array(ag1.positions, ag2.positions, box=ts.dimensions)
        if mode == 'minimum':
            dist[i] = d.min()
        elif mode == 'mean1':
            dist[i] = d.mean(axis=0)
        else:
            dist[i] = d

    if save is not None:
        np.savetxt(save, dist)

    return dist
