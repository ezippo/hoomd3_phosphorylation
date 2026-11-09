"""
Density profiles of centred trajectories (see hpa.trajectory for the centring).

- slab, along one axis: compute_density_profile, compute_density_profile_by_npSer
- droplet, radial: compute_radial_density_profile, compute_radial_density_by_current_npSer
The *_npSer versions split the profile by the number of pSer of each chain.
"""
from collections import defaultdict

import numpy as np
import gsd.hoomd
from tqdm import tqdm


def compute_density_profile(gsd_file, axis=2, nbins=100, group=None, therm=0):
    """
    Compute the density profile along a specified axis for a centered trajectory.

    Parameters
    ----------
    gsd_file : str
        Path to the GSD trajectory file.
    axis : int
        Axis along which to compute the profile: 0=x, 1=y, 2=z.
    nbins : int
        Number of bins along the axis.
    group : list of int or None
        Particle indices to include. If None, include all particles.
    therm : int
        Number of initial frames to exclude for thermalization.

    Returns
    -------
    bin_centers : np.ndarray
        Center coordinates of bins along the axis.
    density : np.ndarray
        Average particle density in each bin.
    """
    traj = gsd.hoomd.open(gsd_file, 'r')[therm:]
    group = np.arange(traj[0].particles.N) if group is None else np.asarray(group)

    # bins over the box along the axis (centred box: [-L/2, L/2])
    box = traj[0].configuration.box[:3]
    L = box[axis]
    edges = np.linspace(-L / 2, L / 2, nbins + 1)

    # histogram frame by frame (positions of all frames are not kept in memory)
    counts = np.zeros(nbins)
    for frame in traj:
        counts += np.histogram(frame.particles.position[group, axis], bins=edges)[0]

    # number density: counts per bin volume (bin width x box area perpendicular to the axis), per frame
    area = np.prod(np.delete(box, axis))
    density = counts / (len(traj) * area * (edges[1] - edges[0]))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return bin_centers, density


def compute_radial_density_profile(gsd_file, nbins=100, r_max=None, group=None, therm=0):
    """
    Compute radial density profile for a centered spherical condensate
    in a cubic box.

    Parameters
    ----------
    gsd_file : str
        Path to GSD trajectory (droplet must already be centered at origin).
    nbins : int
        Number of radial bins.
    r_max : float or None
        Maximum radius for profile. If None, use half box length.
    group : array-like or None
        Particle indices to include. If None, include all particles.
    therm : int
        Number of initial frames to skip.

    Returns
    -------
    r_centers : np.ndarray
        Radial bin centers.
    density : np.ndarray
        Radial number density profile.
    """

    traj = gsd.hoomd.open(gsd_file, 'r')
    frame0 = traj[0]

    box = frame0.configuration.box
    Lx, Ly, Lz = box[:3]

    # ---- Check cubic box ----
    if not np.isclose(Lx, Ly) or not np.isclose(Lx, Lz):
        raise ValueError("Box is not cubic. This function assumes cubic box.")

    L = Lx

    # ---- Radial extent ----
    if r_max is None:
        r_max = L / 2.0

    if isinstance(nbins, int):
        bins_l = np.linspace(0, r_max, nbins + 1)
    elif isinstance(nbins, list) or isinstance(nbins, np.ndarray):
        bins_l = nbins
    else:
        raise TypeError("nbins is not int or list or numpy array.")
    n_bins = len(bins_l)-1
        
    r_centers = 0.5 * (bins_l[:-1] + bins_l[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (bins_l[1:]**3 - bins_l[:-1]**3)

    density = np.zeros(n_bins)
    frame_count = 0

    # ---- Group selection ----
    if group is not None:
        group = np.array(group)

    # ---- Loop over frames ----
    for frame in tqdm(traj[therm:], desc="Computing radial density"):

        pos = frame.particles.position

        if group is not None:
            pos = pos[group]

        # radial distance from origin
        r = np.linalg.norm(pos, axis=1)

        hist, _ = np.histogram(r, bins=bins_l)
        density += hist

        frame_count += 1

    # ---- Normalize ----
    density /= (frame_count * shell_volumes)

    return r_centers, density


def compute_density_profile_by_npSer(gsd_file, n_chains=200, beads_per_chain=154, axis=2, nbins=100, therm=0):
    """
    Compute the density profile along a specified axis for a centered condensate trajectory,
    splitting contributions by chains with different numbers of phospho-serine beads.

    Assumptions
    -----------
    - Trajectory is already centered (condensate COM at z = 0)
    - Box is centered at the origin: axis in [-Lz/2, Lz/2)
    - Chains are stored contiguously in the particle list
    - All chains have the same number of beads
    - Phospho-serine beads have type name "SEP"

    Parameters
    ----------
    gsd_file : str
        Path to the centered GSD trajectory.
    n_chains : int
        Number of polymer chains.
    beads_per_chain : int
        Number of beads per chain.
    axis : int
        Number of the axis of the slab (0->'x' , 1->'y' , 2->'z').
    nbins : int
        Number of bins along z.
    therm : int
        Number of initial frames to discard (thermalization).

    Returns
    -------
    z_centers : np.ndarray, shape (nbins,)
        Bin centers along z.
    density_profiles : dict
        density_profiles[n_pSer] -> number density profile along z.
    """

    traj = gsd.hoomd.open(gsd_file, 'r')
    frame0 = traj[0]

    N = frame0.particles.N
    box = frame0.configuration.box
    Lz = box[axis]
    zmin, zmax = -Lz / 2, Lz / 2

    # ---- binning ----
    bins = np.linspace(zmin, zmax, nbins + 1)
    z_centers = 0.5 * (bins[:-1] + bins[1:])
    dz = bins[1] - bins[0]

    # ---- determine chain phospho content ----
    chain_ids = np.repeat(np.arange(n_chains), beads_per_chain)
    types = frame0.particles.types
    typeid = frame0.particles.typeid[:n_chains*beads_per_chain]
    pser_type_id = types.index("SEP")
    is_phospho = (typeid == pser_type_id)

    # ---- count phospho beads per chain ----
    phospho_per_chain = {
        c: np.sum(is_phospho[chain_ids == c])
        for c in range(n_chains)
    }

    # ---- map each particle to its chain class ----
    particle_class = np.array(
        [phospho_per_chain[c] for c in chain_ids]
    )

    # ---- storage ----
    density_profiles = defaultdict(lambda: np.zeros(nbins))
    frame_count = 0

    # ---- loop over frames ----
    for iframe, frame in enumerate(tqdm(traj, desc="Computing density profiles")):
        if iframe < therm:
            continue

        z = frame.particles.position[:n_chains*beads_per_chain, axis]

        for n_pSer in np.unique(particle_class):
            mask = particle_class == n_pSer
            hist, _ = np.histogram(z[mask], bins=bins)
            density_profiles[n_pSer] += hist

        frame_count += 1

    # ---- normalize to number density ----
    if axis == 0:
        area = box[1] * box[2]
    elif axis == 1:
        area = box[0] * box[2]
    else:
        area = box[0] * box[1]
        
    for n_pSer in density_profiles:
        density_profiles[n_pSer] /= (frame_count * area * dz)
    
    # ---- compute number of chains per class ----
    chain_counts = {}
    for n_pSer in np.unique(particle_class):
        n_particles = np.sum(particle_class == n_pSer)
        chain_counts[n_pSer] = n_particles // beads_per_chain

    return z_centers, dict(density_profiles), chain_counts


def compute_radial_density_by_current_npSer(gsd_file, n_chains=200, beads_per_chain=154, r_max=None, nbins=100, therm=0):
    """
    Compute radial density profiles for a centered spherical condensate,
    split by the instantaneous number of phospho-serine beads per chain.

    Unlike compute_radial_density_by_npSer, the class of every chain is
    recomputed in every frame, so the profiles stay correct when the
    phosphorylation changes during the trajectory (active droplets).

    Parameters
    ----------
    gsd_file : str
        Path to GSD trajectory (droplet already centered at origin).
    n_chains : int
        Total number of polymer chains.
    beads_per_chain : int
        Number of beads per chain.
    r_max : float or None
        Maximum radius for the profile. If None, use half the smallest box length.
    nbins : int
        Number of radial bins.
    therm : int
        Number of initial frames to skip.

    Returns
    -------
    r_centers : np.ndarray
        Radial bin centers.
    density_profiles : dict
        {n_pSer: radial density profile}, averaged over all analyzed frames.
        A chain contributes to class n_pSer only in the frames where it has
        exactly n_pSer pSer beads.
    chain_counts : dict
        {n_pSer: average number of chains in that class per frame}
    """
    traj = gsd.hoomd.open(gsd_file, 'r')
    box = traj[0].configuration.box
    if r_max is None:
        r_max = 0.5 * min(box[:3])

    bins = np.linspace(0, r_max, nbins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    shell_volumes = (4 / 3) * np.pi * (bins[1:]**3 - bins[:-1]**3)

    n_substrate = n_chains * beads_per_chain
    chain_ids = np.repeat(np.arange(n_chains), beads_per_chain)

    density_profiles = defaultdict(lambda: np.zeros(nbins))
    chain_counts_total = defaultdict(float)
    frame_count = 0

    for frame in tqdm(traj[therm:], desc="Computing radial densities"):
        is_phospho = frame.particles.typeid[:n_substrate] == frame.particles.types.index("SEP")
        phospho_per_chain = np.bincount(chain_ids, weights=is_phospho.astype(np.int32), minlength=n_chains).astype(int)
        particle_class = phospho_per_chain[chain_ids]

        classes, counts = np.unique(phospho_per_chain, return_counts=True)
        for n_pSer, count in zip(classes, counts):
            chain_counts_total[n_pSer] += count

        # radial distance from origin
        r = np.linalg.norm(frame.particles.position[:n_substrate], axis=1)
        for n_pSer in classes:
            hist, _ = np.histogram(r[particle_class == n_pSer], bins=bins)
            density_profiles[n_pSer] += hist

        frame_count += 1

    # ---- normalize by shell volume and frames ----
    for n_pSer in density_profiles:
        density_profiles[n_pSer] /= (frame_count * shell_volumes)

    chain_counts = {n_pSer: count / frame_count for n_pSer, count in chain_counts_total.items()}

    return r_centers, dict(density_profiles), chain_counts
