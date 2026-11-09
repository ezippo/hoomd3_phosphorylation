"""
Markov state models (MSMs) of the enzymatic phosphorylation cycle.

The module covers two steps:

1. State assignment: turn distance and phosphorylation time series into
   discrete state trajectories (bound/unbound x phosphorylated/unphosphorylated).
2. Estimation: transition matrices, cycle affinities (Delta mu), edge
   contributions, block-bootstrap errors and Chapman-Kolmogorov tests.

Convention: all estimation functions take *0-based* state trajectories
(state 1 of `create_states_trajectory` is index 0).
"""
import numpy as np
import deeptime.markov as markov

KB = 0.00831446  # Boltzmann constant [kJ/(mol K)]


# ---------------------------------------------------------------------------
# State assignment
# ---------------------------------------------------------------------------

def _stretches(a, value):
    """Return (start, end) index pairs of the consecutive stretches where a == value.

    `end` is the first index after the stretch.
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    """
    is_value = np.concatenate(([0], np.equal(a, value).view(np.int8), [0]))
    return np.flatnonzero(np.abs(np.diff(is_value)) == 1).reshape(-1, 2)


def _split_transition_paths(ar, tps_idx, verbose=False):
    """Assign the second half of each transition path to the state that follows it.

    The first half is longer for odd-length paths.
    """
    for tp_start, tp_end in tps_idx:
        second_half = tp_start + (tp_end - tp_start + 1) // 2
        if verbose:
            print('start of second half of transition path, end transition path')
            print(second_half, tp_end)
        ar[second_half:tp_end] = ar[tp_end]
    return ar


def transition_filter_state_trj(cba_ar, split_state_tp=True, return_trans_start_end=False,
                                tps_state_num=0, verbose=False):
    """Transition-based assignment: remove the transition-path state `tps_state_num`.

    Frames in a transition path are given to the previous stable state; with
    `split_state_tp` the second half of each path is given to the next stable state.
    A trajectory that starts or ends in a transition path borrows the first/last stable state.

    Returns the filtered trajectory and, if `return_trans_start_end`, the
    (start, end) indices of the transition paths.
    """
    tba_ar = np.array(cba_ar, copy=True)

    not_tps = tba_ar[tba_ar != tps_state_num]
    if tba_ar[-1] == tps_state_num:
        tba_ar[-1] = not_tps[-1]
    if tba_ar[0] == tps_state_num:
        tba_ar[0] = not_tps[0]

    tps_idx = _stretches(tba_ar, tps_state_num)
    for tp_start, tp_end in tps_idx:
        tba_ar[tp_start:tp_end] = tba_ar[tp_start - 1]

    if split_state_tp:
        tba_ar = _split_transition_paths(tba_ar, tps_idx, verbose)

    if return_trans_start_end:
        return tba_ar, tps_idx
    return tba_ar


def inserting_tps_state(boundtraj, dist, min_dist, max_dist=None, tps_state_num=None):
    """Mark frames with min_dist < dist < max_dist as the transition-path state.

    If `max_dist` is None the trajectory is returned unchanged.
    `tps_state_num` defaults to max(boundtraj) + 1.
    """
    if max_dist is None:
        return boundtraj
    if tps_state_num is None:
        tps_state_num = np.max(boundtraj) + 1

    in_between = (dist > min_dist) & (dist < max_dist)
    return np.where(in_between, tps_state_num, boundtraj)


def boundtraj_with_dist_criterion(dist, min_dist=10., max_dist=None, therm=0, end_time=None, save=None):
    """Bound/unbound trajectory from a distance array of shape (n_frames, n_beads).

    The enzyme is bound when the minimum distance over the beads is < min_dist.
    Frames with min_dist < distance < max_dist form transition paths that are
    resolved with `transition_filter_state_trj`.
    """
    mindistances = np.min(dist, axis=1)
    boundbool = mindistances < min_dist
    boundbool = inserting_tps_state(boundbool, mindistances, min_dist, max_dist=max_dist, tps_state_num=2)

    filtered = transition_filter_state_trj(boundbool, tps_state_num=2).astype(bool)[therm:end_time]

    if save is not None:
        np.savetxt(save, filtered)
    return filtered


def changes_to_phosphostate(changes, start_time=0, step=10000, end_time=3000000000, save=None):
    """Phosphorylation state (0/1) at every dump step from the list of accepted changes.

    Args:
        changes (ndarray): rows of accepted (de)phosphorylation events, sorted in
            time; column 0 is the MD step, column 2 the event code (1 or 10 for
            phosphorylation). Rejected attempts must be removed beforehand.
        start_time (int): initial MD step.
        step (int): dump interval in MD steps.
        end_time (int): final MD step.
        save (str, optional): file to save the trajectory to.

    Frame t holds the state after all changes up to start_time + (t+1)*step.
    """
    if changes[0, 2] in (1, 10):
        boolstart = 0
    else:
        print(f'start with pSer: changes[0] = {changes[0, 2]}')
        boolstart = 1

    n_frames = int((end_time - start_time) / step)
    frame_ends = start_time + (np.arange(n_frames) + 1) * step
    n_changes = np.searchsorted(changes[:, 0], frame_ends, side='right')
    phosphostate = (boolstart + n_changes) % 2

    if save is not None:
        np.savetxt(save, phosphostate)
    return phosphostate


def create_states_trajectory(boundtraj, phosphotraj, save=None):
    """4-state trajectory from binding and phosphorylation boolean trajectories.

    States:
        1: unphosphorylated, unbound
        2: unphosphorylated, bound
        3: phosphorylated, bound
        4: phosphorylated, unbound
    """
    bound = np.asarray(boundtraj, dtype=bool)
    phospho = np.asarray(phosphotraj, dtype=bool)

    states = np.zeros(len(phospho), dtype=int)
    states[~phospho & ~bound] = 1
    states[~phospho & bound] = 2
    states[phospho & bound] = 3
    states[phospho & ~bound] = 4

    if save is not None:
        np.savetxt(save, states, fmt='%d')
    return states


def create_states_trajectory_2enzymes(boundtraj_1, boundtraj_2, phosphotraj, save=None):
    """8-state trajectory for two enzymes (1 = kinase, 2 = phosphatase).

    States (P = phosphorylated, U = unphosphorylated):
        1: U, free          2: U, enz1 bound    3: P, enz1 bound
        4: P, free          5: P, enz2 bound    6: U, enz2 bound
        7: U, both bound    8: P, both bound
    States 1-6 form the cycle 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1.
    """
    b1 = np.asarray(boundtraj_1, dtype=bool)
    b2 = np.asarray(boundtraj_2, dtype=bool)
    phospho = np.asarray(phosphotraj, dtype=bool)

    states = np.zeros(len(phospho), dtype=int)
    states[~phospho & ~b1 & ~b2] = 1
    states[~phospho & b1 & ~b2] = 2
    states[~phospho & ~b1 & b2] = 6
    states[~phospho & b1 & b2] = 7
    states[phospho & ~b1 & ~b2] = 4
    states[phospho & b1 & ~b2] = 3
    states[phospho & ~b1 & b2] = 5
    states[phospho & b1 & b2] = 8

    if save is not None:
        np.savetxt(save, states, fmt='%d')
    return states


def concatenate_trajectories(trajs):
    """Join state trajectories of independent simulations into a single one.

    Each trajectory after the first starts at its first frame in the state in
    which the previous trajectory ended, so that no artificial jump is
    created at the junction.
    """
    pieces = [np.asarray(trajs[0])]
    for traj in trajs[1:]:
        traj = np.asarray(traj)
        start = np.flatnonzero(traj == pieces[-1][-1])[0]
        pieces.append(traj[start:])
    return np.concatenate(pieces)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def transition_matrix(dtraj, lag, n_states=None, reversible=False):
    """Maximum-likelihood transition matrix of a 0-based state trajectory.

    The MSM is estimated on the largest connected set; rows and columns of
    states outside that set are NaN, so indices always match the state labels.
    """
    counts = markov.TransitionCountEstimator(lagtime=lag, count_mode='effective',
                                             n_states=n_states).fit_fetch(dtraj)
    msm = markov.msm.MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)

    tmat = np.full((counts.n_states_full, counts.n_states_full), np.nan)
    active = msm.count_model.state_symbols
    tmat[np.ix_(active, active)] = msm.transition_matrix
    return tmat


def _cycle_edges(n_states, cycle=None):
    """Forward edges (i, j) of a closed cycle; default cycle is 0 -> 1 -> ... -> n-1 -> 0."""
    cycle = list(range(n_states)) if cycle is None else list(cycle)
    if cycle[0] != cycle[-1]:
        cycle.append(cycle[0])
    return list(zip(cycle[:-1], cycle[1:]))


def edge_dmus(tmat, kT, cycle=None):
    """Contribution kT * ln(T_ij / T_ji) of every edge (i, j) of the cycle."""
    return np.array([kT * np.log(tmat[i, j] / tmat[j, i]) for i, j in _cycle_edges(len(tmat), cycle)])


def cycle_dmu(tmat, kT, cycle=None):
    """Cycle affinity kT * ln(prod T_forward / prod T_backward)."""
    edges = _cycle_edges(len(tmat), cycle)
    forward = np.prod([tmat[i, j] for i, j in edges])
    backward = np.prod([tmat[j, i] for i, j in edges])
    return kT * np.log(forward / backward)


def block_resample(dtraj, pow_bin, n_resample, rng=None):
    """Yield block-bootstrap resamples of a trajectory (block size 2**pow_bin frames).

    Use blocks much longer than the lag time: each junction between blocks
    introduces artificial transitions.
    """
    rng = np.random.default_rng(rng)
    block = 2**pow_bin
    n_blocks = len(dtraj) // block
    blocks = np.asarray(dtraj)[:n_blocks * block].reshape(n_blocks, block)
    for _ in range(n_resample):
        yield blocks[rng.integers(n_blocks, size=n_blocks)].ravel()


def edge_dmus_bootstrap(dtraj, lag, kT, cycle=None, pow_bin=10, n_resample=30, reversible=False, rng=None):
    """Edge contributions to the cycle affinity and their block-bootstrap standard deviations."""
    n_states = np.max(dtraj) + 1
    edges = edge_dmus(transition_matrix(dtraj, lag, n_states, reversible), kT, cycle)
    samples = [edge_dmus(transition_matrix(d, lag, n_states, reversible), kT, cycle)
               for d in block_resample(dtraj, pow_bin, n_resample, rng)]
    return edges, np.nanstd(samples, axis=0)


def dmu_with_error(dtraj, lag, kT, cycle=None, pow_bin=10, n_resample=30, reversible=False, rng=None):
    """Cycle affinity with error propagated from bootstrapped transition probabilities.

    The block-bootstrap standard deviation of every T_ij on the cycle is
    propagated assuming independent elements:
        d(dmu) = kT * sqrt(sum_edges (dT_ij/T_ij)^2 + (dT_ji/T_ji)^2)
    """
    n_states = np.max(dtraj) + 1
    tmat = transition_matrix(dtraj, lag, n_states, reversible)
    samples = [transition_matrix(d, lag, n_states, reversible)
               for d in block_resample(dtraj, pow_bin, n_resample, rng)]
    dtmat = np.nanstd(samples, axis=0)

    edges = _cycle_edges(n_states, cycle)
    rel_err = [dtmat[i, j] / tmat[i, j] for i, j in edges] + [dtmat[j, i] / tmat[j, i] for i, j in edges]
    return cycle_dmu(tmat, kT, cycle), kT * np.sqrt(np.sum(np.square(rel_err)))


def ck_test(dtraj, lagtimes, base_lag=1, reversible=False, n_samples=100, n_steps=20):
    """Chapman-Kolmogorov test of the Bayesian MSM estimated at `base_lag`.

    Returns, for every lag time in `lagtimes` (multiples of `base_lag`):
        predicted: T(base_lag)^(lag/base_lag)
        estimated: mean of the Bayesian MSM estimated at that lag
        estimated_err: standard deviation of the Bayesian MSM samples
    """
    def bayesian_tmat(lag):
        counts = markov.TransitionCountEstimator(lagtime=lag, count_mode='effective').fit_fetch(dtraj)
        msm = markov.msm.BayesianMSM(n_samples=n_samples, n_steps=n_steps,
                                     reversible=reversible).fit_fetch(counts)
        stats = msm.gather_stats('transition_matrix')
        return stats.mean, stats.std

    if any(lag % base_lag for lag in lagtimes):
        raise ValueError('all lag times must be multiples of base_lag')

    tmat_base, _ = bayesian_tmat(base_lag)
    predicted, estimated, estimated_err = [], [], []
    for lag in lagtimes:
        mean, std = bayesian_tmat(lag)
        predicted.append(np.linalg.matrix_power(tmat_base, lag // base_lag))
        estimated.append(mean)
        estimated_err.append(std)

    return np.array(predicted), np.array(estimated), np.array(estimated_err)
