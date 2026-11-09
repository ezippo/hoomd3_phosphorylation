# Analysis code

Notebooks that make the figures of Zippo et al., *Kinase-phosphatase cycles drive self-sustained oscillations of protein
condensates* (bioRxiv 2026), from the data of this record, and the `hpa` package they use.

- `*.ipynb`: one notebook per group of figures (see the table in the main `README.md`)
- `src/hpa/`: **HPA** (HPS phosphorylation analysis), functions to analyse HPS/CALVADOS trajectories with
  phosphorylation reactions.
- `src/hpa/paths.py`: locations of the data (set `HPA_DATA_ROOT` to the folder of the record)
- `plot_styles/`: matplotlib style files
- `vampnet_config_PP3_CAL3_tlag10_trained_fr01.json`: settings of the trained VAMPnet

**Requirements**: Python 3.11 with numpy, scipy, matplotlib, gsd, freud, MDAnalysis, deeptime, networkx, tqdm.

Run the notebooks from this folder (they import `hpa` from `src/`):

```bash
export HPA_DATA_ROOT=/path/to/this/record
jupyter lab
```

The package was first introduced with

Zippo, E., Dormann, D., Speck, T., Stelzl, L.S. Molecular simulations of enzymatic phosphorylation of disordered
proteins and their condensates. Nat Commun 16, 4649 (2025). https://doi.org/10.1038/s41467-025-59676-4

Licence: GPL-3.0 (`LICENSE`).
