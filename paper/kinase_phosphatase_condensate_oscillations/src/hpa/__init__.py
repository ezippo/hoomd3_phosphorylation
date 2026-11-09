"""
HPA: analysis of HPS/CALVADOS simulations with phosphorylation reactions.

Modules (import the one you need, e.g. `import hpa.timelines as hpt`):
    cluster     size of the condensate and pSer in time (from dumps or size files)
    contacts    residue-residue contact maps between chains
    msm         Markov state models of the enzymatic cycle: states, transition matrices, cycle affinities
    plots       plots shared by the notebooks
    profiles    slab and radial density profiles of centred trajectories
    rates       phosphorylation reactions: counts per serine, pSer per chain
    stats       fit models, error estimates, cross-correlations
    timelines   A/B x in/out chain-state timelines, transition rates and lifetimes
    trajectory  PBC helpers, centring of trajectories, distance time series

Usage from a notebook:
    import sys
    sys.path.insert(0, '/localscratch/zippoema/git/paper_fuel-driven_reactions_condensate/src/')
    import hpa.timelines as hpt
"""
