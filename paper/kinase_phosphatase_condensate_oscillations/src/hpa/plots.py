"""
Plots shared by the notebooks.

- plot_dist: distance time series (e.g. from hpa.trajectory.create_distance_file)
- plot_radial_density_profiles_by_npSer: profiles from hpa.profiles.*_npSer
- plot_fading_trajectory: trajectories in a plane fading in time (limit cycles)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection


def plot_dist(dist, mindist=False, part_id=0, start=0, end=None, jump=1, step_in_ps=100, fig_len=6, fig_hig=2.5,
              linewidth=0.4, ylabel=r"dist(SER, CK1$\delta$)"):
    """Distance vs time (dist: frames x particles); column part_id, or the minimum over particles if mindist."""
    if end==None:
        end_tmp = len(dist)
    else:
        end_tmp = end
    if mindist:
        d = np.min(dist, axis=1)[start:end_tmp:jump]
    else:
        d = dist[start:end_tmp:jump,part_id]
        
    time = np.arange(start, end_tmp, jump)*step_in_ps/1000000
    
    plt.figure(figsize=(fig_len,fig_hig))
    plt.plot(time, d, linewidth=linewidth)
    plt.xlim(step_in_ps*(start-1)/1000000, step_in_ps*(1+end_tmp)/1000000)
    plt.xlabel(r't [$\mu$s]')
    plt.ylabel(ylabel)
    plt.show()


def plot_radial_density_profiles_by_npSer(
    r_centers,
    density_profiles,
    chain_counts,
    select=None,
    groups=None,
    cmap="viridis",
    lw=2,
    per_chain=True,
    radial=False,
    logscale=False,
    save=None
):
    """
    Plot radial density profiles split or grouped by phosphorylation state.

    Parameters
    ----------
    r_centers : np.ndarray
        Radial bin centers.
    density_profiles : dict
        {n_pSer: density(r)}.
    chain_counts : dict
        {n_pSer: number of chains in that class}.
    select : list[int] or None
        Plot only specific phosphorylation states (ignored if groups is used).
    groups : dict or None
        Optional grouping rules:
            {label: function(n_pSer) -> bool}
        Example:
            groups = {
                "0–5 pSer": lambda x: 0 <= x <= 5,
                "6–15 pSer": lambda x: 6 <= x <= 15,
                ">15 pSer": lambda x: x > 15
            }
    cmap : str
        Colormap.
    lw : float
        Line width.
    per_chain : bool
        If True, normalize densities per chain.
    radial : bool
        If True, xaxis label is r
    logscale : bool
        If True, yscale is log.

    Returns
    -------
    dict
        Plotted profiles: {group label: density} with `groups`, {n_pSer: density} otherwise.
    """

    # ---------- GROUPED MODE ----------
    if groups is not None:
        merged_profiles = {}
        labels = []

        for label, rule in groups.items():
            matching_classes = [c for c in density_profiles if rule(c)]
            if not matching_classes:
                continue

            total_chains = sum(chain_counts[c] for c in matching_classes)

            if per_chain:
                merged_density = sum(
                    density_profiles[c] for c in matching_classes
                ) / total_chains
            else:
                merged_density = sum(
                    density_profiles[c] for c in matching_classes
                )

            merged_profiles[label] = merged_density
            labels.append(label)

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(labels)))

        plt.figure(figsize=(6, 4))
        for label, color in zip(labels, colors):
            plt.plot(r_centers, merged_profiles[label],
                     color=color, lw=lw, label=label)

    # ---------- INDIVIDUAL CLASS MODE ----------
    else:
        classes = sorted(density_profiles.keys())
        if select is not None:
            classes = [c for c in classes if c in select]

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(classes)))

        merged_profiles = {}
        plt.figure(figsize=(6, 4))
        for c, color in zip(classes, colors):
            rho = density_profiles[c]
            if per_chain:
                rho = rho / chain_counts[c]
            merged_profiles[c] = rho

            plt.plot(r_centers, rho, color=color, lw=lw, label=f"{c} pSer")

    if radial:
        plt.xlabel("r")
    else:
        plt.xlabel("z")
    plt.ylabel("Density per chain")
    if logscale:
        plt.yscale('log')
    plt.legend(title="Phosphorylation", fontsize=9)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save+'.png', dpi=600)
        plt.savefig(save+'.pdf', dpi=600)
    else:
        plt.show()

    return merged_profiles


def plot_fading_trajectory(ax, x, y, color, linewidth=0.8, label=None, therm=0, plot_mean=True, linestyle='-'):
    """Trajectory (x, y) drawn from white to `color` as time goes on (e.g. limit cycles);
    the first `therm` points are skipped. Optional mean +- std marker."""
    xx, yy = x[therm:], y[therm:]
    points = np.array([xx, yy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    base = np.array(to_rgb(color))
    strength = np.linspace(0.15, 1.0, len(segments))      # white -> base color
    ax.plot(xx[-1], yy[-1], color=color, linestyle=linestyle, label=label)
    ax.add_collection(LineCollection(segments, colors=1 - strength[:, None] * (1 - base),
                                     linewidth=linewidth, linestyle=linestyle))
    if plot_mean:
        ax.errorbar(np.mean(xx), np.mean(yy), np.std(yy), xerr=np.std(xx), fmt='o', color=base * 0.6, capsize=4)
