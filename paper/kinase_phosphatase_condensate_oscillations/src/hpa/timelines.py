"""
Chain-state timelines of substrate chains in a condensate.

Every chain is classified as A (n_pSer < threshold) or B (n_pSer >= threshold)
and by its region (in / out of the largest cluster, optionally interface).
A timeline stores, for each chain, the segments (t0, t1, state) with times in
MD steps shifted by the thermalization time.

Pipeline:
    chemical events  (build_chemical_events, from the reaction file)
  + region events    (build_region_events / build_region_events_interface, from the dumps)
  -> timelines        (build_two/four/six_state_timelines)
  -> rates, lifetimes, cycle affinities (compute_rates, kaplan_meier, cycle_dmu_from_rates)
"""
import json
from collections import defaultdict

import numpy as np
import gsd.hoomd
import freud
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def build_chemical_events(gsd_file, reaction_file, n_chains, chain_length, phospho_type='SEP',
                          threshold=6, t_therm=0):
    """A/B changes of every chain from the accepted (de)phosphorylations.

    The initial pSer pattern is read from the first frame of `gsd_file`; events
    up to t_therm only evolve the state.

    Returns
        chemical_events[chain] = [(t - t_therm, 'A' or 'B'), ...]
        initial_state: A/B state of each chain at t_therm
    """
    with gsd.hoomd.open(gsd_file, 'r') as traj:
        snap = traj[0]
        types = snap.particles.types
        typeid = snap.particles.typeid[:n_chains * chain_length]
    phospho_state = (np.asarray(types)[typeid] == phospho_type).astype(np.int8)

    bead_to_chain = np.arange(n_chains * chain_length) // chain_length
    chain_counts = np.bincount(bead_to_chain, weights=phospho_state, minlength=n_chains).astype(np.int32)

    def classify(c):
        return 'A' if c < threshold else 'B'

    current_state = np.array([classify(c) for c in chain_counts], dtype='<U1')
    initial_state = current_state.copy()

    data = np.loadtxt(reaction_file, usecols=(0, 1, 2), dtype=float)
    times = data[:, 0].astype(int)
    beads = data[:, 1].astype(int)
    changes = data[:, 2].astype(int)
    valid = (changes == 1) | (changes == -1)

    chemical_events = {c: [] for c in range(n_chains)}
    for t, bead, change in zip(times[valid], beads[valid], changes[valid]):
        chain = bead_to_chain[bead]
        phospho_state[bead] += change
        if phospho_state[bead] not in (0, 1):
            raise ValueError(f'Invalid phosphorylation state at bead {bead}: '
                             f'{phospho_state[bead]} (after change {change} at time {t})')

        chain_counts[chain] += change
        new_state = classify(chain_counts[chain])
        current_state[chain] = new_state

        if t <= t_therm:
            initial_state = current_state.copy()
        else:
            chemical_events[chain].append((t - t_therm, new_state))

    return chemical_events, initial_state


def build_region_events(dump_files, n_chains, chain_length, cutoff=1.0, frac_threshold=0.5, t_therm=0):
    """in/out changes of every chain: 'in' if more than `frac_threshold` of its beads are in the largest cluster.

    Returns
        region_events[chain] = [(t - t_therm, 'in' or 'out'), ...]
        initial_regions: region of each chain in the first frame at or after t_therm
    """
    n_substrate = n_chains * chain_length
    region_events = {c: [] for c in range(n_chains)}
    current_region = np.empty(n_chains, dtype='<U3')
    initial_regions = np.empty(n_chains, dtype='<U3')

    initialized = False
    for f in dump_files:
        traj = gsd.hoomd.open(f, 'r')
        box = freud.box.Box.from_box(traj[0].configuration.box)
        cluster = freud.cluster.Cluster()

        for snap in tqdm(traj):
            t = snap.configuration.step
            if t < t_therm:
                continue

            cluster.compute((box, snap.particles.position[:n_substrate]), neighbors={'r_max': cutoff})
            cluster_ids = cluster.cluster_idx
            largest = np.argmax(np.bincount(cluster_ids))
            frac_in = np.mean(cluster_ids.reshape(n_chains, chain_length) == largest, axis=1)
            new_region = np.where(frac_in > frac_threshold, 'in', 'out')

            if not initialized:
                initial_regions[:] = new_region
                current_region[:] = new_region
                initialized = True
                continue

            changed = new_region != current_region
            for c in np.flatnonzero(changed):
                region_events[c].append((t - t_therm, new_region[c]))
            current_region[changed] = new_region[changed]

    return region_events, initial_regions


def classify_regions_radial(positions, box, n_chains, chain_length, cluster_r_cut=1.0, interface_thickness=2.0,
                            min_interface_fraction=0.2, radius_percentile=95):
    """Classify each substrate chain as 'out', 'in' or 'interface'.

        'out'       : no bead in the largest cluster
        'in'        : some beads in the largest cluster, few of them at the interface
        'interface' : at least `min_interface_fraction` of its cluster beads lie in the
                      radial shell R +- interface_thickness, with R the `radius_percentile`
                      of the cluster-bead distances from the (periodic) cluster COM

    Returns chain_region, bead_region (0 out, 1 in, 2 interface), n_cluster,
    interface_fraction, condensate_com, condensate_radius.
    """
    n_substrate = n_chains * chain_length
    positions = positions[:n_substrate]

    cluster = freud.cluster.Cluster()
    cluster.compute((box, positions), neighbors={'r_max': cluster_r_cut})
    cluster_idx = np.asarray(cluster.cluster_idx)
    in_cluster = cluster_idx == np.argmax(np.bincount(cluster_idx))
    cluster_indices = np.flatnonzero(in_cluster)
    cluster_positions = positions[in_cluster]

    if len(cluster_positions) == 0:
        return (np.full(n_chains, 'out', dtype='<U9'), np.zeros(n_substrate, dtype=np.int8),
                np.zeros(n_chains), np.zeros(n_chains), None, 0.0)

    # condensate COM under PBC (circular mean of fractional coordinates)
    angles = 2.0 * np.pi * (box.make_fractional(cluster_positions) % 1.0)
    fractional_com = (np.arctan2(np.mean(np.sin(angles), axis=0), np.mean(np.cos(angles), axis=0)) / (2.0 * np.pi)) % 1.0
    condensate_com = box.make_absolute(fractional_com)

    r_cluster = np.linalg.norm(box.wrap(cluster_positions - condensate_com), axis=1)
    condensate_radius = np.percentile(r_cluster, radius_percentile)
    r_inner = max(0.0, condensate_radius - interface_thickness)
    r_outer = condensate_radius + interface_thickness

    bead_region = np.zeros(n_substrate, dtype=np.int8)
    bead_region[cluster_indices] = 1
    bead_region[cluster_indices[(r_cluster >= r_inner) & (r_cluster <= r_outer)]] = 2

    region_2d = bead_region.reshape(n_chains, chain_length)
    n_cluster = np.sum(region_2d > 0, axis=1)
    n_interface = np.sum(region_2d == 2, axis=1)
    has_cluster = n_cluster > 0
    interface_fraction = np.zeros(n_chains)
    interface_fraction[has_cluster] = n_interface[has_cluster] / n_cluster[has_cluster]

    chain_region = np.full(n_chains, 'out', dtype='<U9')
    chain_region[has_cluster] = 'in'
    chain_region[has_cluster & (interface_fraction >= min_interface_fraction)] = 'interface'

    return chain_region, bead_region, n_cluster, interface_fraction, condensate_com, condensate_radius


def build_region_events_interface(gsd_files, n_chains, chain_length, t_therm=0, t_end=None, **classify_kwargs):
    """out/interface/in changes of every chain (see classify_regions_radial for the keyword arguments).

    Returns region_events, initial_regions and the end time shifted by t_therm.
    """
    if isinstance(gsd_files, str):
        gsd_files = [gsd_files]

    region_events = {chain: [] for chain in range(n_chains)}
    current_regions = None
    initial_regions = None
    last_timestep = None

    for gsd_file in gsd_files:
        with gsd.hoomd.open(gsd_file, 'r') as traj:
            for frame in tqdm(traj):
                timestep = int(frame.configuration.step)
                last_timestep = timestep
                if timestep < t_therm:
                    continue
                if t_end is not None and timestep > t_end:
                    break

                chain_region = classify_regions_radial(np.asarray(frame.particles.position),
                                                       freud.box.Box.from_box(frame.configuration.box),
                                                       n_chains, chain_length, **classify_kwargs)[0]
                if current_regions is None:
                    current_regions = chain_region.copy()
                    initial_regions = chain_region.copy()
                    continue

                for chain in np.flatnonzero(chain_region != current_regions):
                    region_events[chain].append((timestep - t_therm, chain_region[chain]))
                current_regions = chain_region.copy()

        if t_end is not None and last_timestep >= t_end:
            break

    if initial_regions is None:
        raise ValueError('No GSD snapshot found at or after t_therm.')

    t_end_shifted = (last_timestep if t_end is None else t_end) - t_therm
    return region_events, initial_regions, t_end_shifted


# ---------------------------------------------------------------------------
# Timelines
# ---------------------------------------------------------------------------

def merge_identical_segments(timelines):
    """Merge consecutive segments with the same state."""
    merged_timelines = {}
    for chain, segments in timelines.items():
        merged = []
        for t0, t1, state in segments:
            if merged and merged[-1][2] == state:
                merged[-1] = (merged[-1][0], t1, state)
            else:
                merged.append((t0, t1, state))
        merged_timelines[chain] = merged
    return merged_timelines


def build_two_state_timelines(chemical_events, initial_state, n_chains, t_end):
    """A/B timelines (consecutive segments with the same state are merged)."""
    timelines = {}
    for chain in range(n_chains):
        segments = []
        last_t = 0
        current_state = initial_state[chain]
        for t, new_state in chemical_events[chain]:
            if t > last_t:
                segments.append((last_t, t, current_state))
            current_state = new_state
            last_t = t
        if t_end > last_t:
            segments.append((last_t, t_end, current_state))
        timelines[chain] = segments
    return merge_identical_segments(timelines)


def _combine_events(chemical_events, region_events, initial_AB, initial_regions, n_chains, t_end):
    """Merge the two sorted event streams into '{AB}_{region}' timelines."""
    timelines = {}
    for chain in range(n_chains):
        chem = chemical_events[chain]
        reg = region_events[chain]
        current_AB = initial_AB[chain]
        current_region = initial_regions[chain]
        current_state = f'{current_AB}_{current_region}'

        last_t = 0
        segments = []
        i = j = 0
        while i < len(chem) or j < len(reg):
            # chemical event first when both happen at the same time
            if j >= len(reg) or (i < len(chem) and chem[i][0] <= reg[j][0]):
                t, current_AB = chem[i]
                i += 1
            else:
                t, current_region = reg[j]
                j += 1
            t = int(t)
            if t > last_t:
                segments.append((last_t, t, current_state))
            current_state = f'{current_AB}_{current_region}'
            last_t = t

        if t_end > last_t:
            segments.append((last_t, int(t_end), current_state))
        timelines[int(chain)] = segments

    return merge_identical_segments(timelines)


def build_four_state_timelines(chemical_events, region_events, initial_AB, initial_regions, n_chains, t_end):
    """Timelines with states A_in, A_out, B_in, B_out."""
    return _combine_events(chemical_events, region_events, initial_AB, initial_regions, n_chains, t_end)


def build_six_state_timelines(chemical_events, region_events, initial_AB, initial_regions, n_chains, t_end):
    """Timelines with states A/B x in/interface/out."""
    return _combine_events(chemical_events, region_events, initial_AB, initial_regions, n_chains, t_end)


def save_timelines(timelines, filename):
    with open(filename, 'w') as f:
        json.dump(timelines, f, indent=4)


def load_timelines(filename):
    """Load timelines saved as json (keys back to int, segments to tuples)."""
    with open(filename) as f:
        tl = json.load(f)
    return {int(chain): [tuple(s) for s in segments] for chain, segments in tl.items()}


def merge_replicas(timelines_list, n_chains):
    """Join the timelines of several replicas, renumbering the chains of replica k as chain + k*n_chains."""
    merged = {}
    for k, tl in enumerate(timelines_list):
        for chain, segments in tl.items():
            merged[int(chain) + k * n_chains] = segments
    return merged


def restrict_timelines(timelines, t0, t1):
    """Clip all segments to the interval [t0, t1]."""
    if t1 <= t0:
        raise ValueError('t1 must be larger than t0.')
    restricted = {}
    for chain, segments in timelines.items():
        restricted[chain] = [(max(start, t0), min(end, t1), state) for start, end, state in segments
                             if end > t0 and start < t1 and min(end, t1) > max(start, t0)]
    return restricted


def region_timelines(timelines):
    """in/out (or interface) timelines from '{AB}_{region}' timelines."""
    return merge_identical_segments({chain: [(t0, t1, state.split('_')[1]) for t0, t1, state in segments]
                                     for chain, segments in timelines.items()})


def states_at_frames(timelines, frame_times, n_chains):
    """State of every chain at the given (shifted) times: segment with t0 <= t < t1."""
    states = np.empty((len(frame_times), n_chains), dtype='<U12')
    for c in range(n_chains):
        segs = timelines[c] if c in timelines else timelines[str(c)]
        starts = np.array([s[0] for s in segs])
        labels = np.array([s[2] for s in segs])
        idx = np.searchsorted(starts, frame_times, side='right') - 1
        states[:, c] = labels[np.clip(idx, 0, len(segs) - 1)]
    return states


def compute_state_populations(timelines, snapshot_times, states=('A_in', 'A_out', 'B_in', 'B_out')):
    """Number of chains in every state at each snapshot time."""
    populations = {state: np.zeros(len(snapshot_times), dtype=int) for state in states}
    for segments in timelines.values():
        seg_idx = 0
        for snap_idx, t in enumerate(snapshot_times):
            while seg_idx < len(segments) - 1 and t >= segments[seg_idx][1]:
                seg_idx += 1
            populations[segments[seg_idx][2]][snap_idx] += 1
    return populations


# ---------------------------------------------------------------------------
# Rates and lifetimes
# ---------------------------------------------------------------------------

def compute_rates(timelines):
    """Transition rates n_ij / T_i with Poisson errors sqrt(n_ij) / T_i.

    T_i is the total residence time in state i (including the last, censored segment).
    Returns rates, errors, counts, residence (dicts keyed by (state_i, state_j) or state_i).
    """
    counts = defaultdict(int)
    residence = defaultdict(float)
    for segments in timelines.values():
        for i, (t0, t1, state) in enumerate(segments):
            residence[state] += t1 - t0
            if i < len(segments) - 1 and segments[i + 1][2] != state:
                counts[(state, segments[i + 1][2])] += 1

    rates, errors = {}, {}
    for (s0, s1), n in counts.items():
        T = residence[s0]
        rates[(s0, s1)] = n / T if T > 0 else np.nan
        errors[(s0, s1)] = (np.sqrt(n) / T if n > 0 else 0.0) if T > 0 else np.nan
    return rates, errors, counts, residence


def cycle_dmu_from_rates(rates, cycle, kT):
    """kT * ln(prod forward rates / prod backward rates) along the closed cycle of states."""
    cycle = list(cycle)
    edges = list(zip(cycle, cycle[1:] + cycle[:1]))
    forward = np.prod([rates[(i, j)] for i, j in edges])
    backward = np.prod([rates[(j, i)] for i, j in edges])
    return kT * np.log(forward / backward)


def net_fluxes(counts, t_total, n_chains):
    """Net transitions per chain per MD step, (n_ij - n_ji) / (t_total * n_chains), for every observed pair."""
    return {(i, j): (counts.get((i, j), 0) - counts.get((j, i), 0)) / t_total / n_chains for i, j in counts}


def extract_lifetimes(timelines):
    """Segment durations per state; the last segment of each chain is right-censored."""
    states = sorted({state for segments in timelines.values() for _, _, state in segments})
    print(f'states : {states}')
    lifetimes = {s: [] for s in states}
    censored = {s: [] for s in states}
    for segments in timelines.values():
        for i, (t0, t1, state) in enumerate(segments):
            (censored if i == len(segments) - 1 else lifetimes)[state].append(t1 - t0)
    return ({s: np.array(v) for s, v in lifetimes.items()}, {s: np.array(v) for s, v in censored.items()})


def kaplan_meier(lifetimes, censored):
    """Kaplan-Meier survival function from observed and right-censored lifetimes."""
    times = np.concatenate([lifetimes, censored])
    events = np.concatenate([np.ones_like(lifetimes), np.zeros_like(censored)])
    order = np.argsort(times)
    times, events = times[order], events[order]

    n_at_risk = len(times)
    survival = 1.0
    S = []
    for e in events:
        if e == 1:
            survival *= 1 - 1 / n_at_risk
        S.append(survival)
        n_at_risk -= 1
    return times, np.array(S)
