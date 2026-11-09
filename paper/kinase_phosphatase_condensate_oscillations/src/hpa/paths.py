"""
Locations of the data used by the notebooks.

DATA_ROOT is the folder that contains simulations/, analysis/ and plots/ (the unpacked Zenodo record, or the
original project folder). Set it with the environment variable HPA_DATA_ROOT, e.g.

    export HPA_DATA_ROOT=/path/to/zenodo_record

before starting Jupyter; without it the original location on the workstation is used.

All paths are strings ending in '/', so that the notebooks can write SIM + 'droplet_200-tdp43-lcd/'.
"""
import os
from pathlib import Path

DATA_ROOT = os.environ.get('HPA_DATA_ROOT', '/localscratch/zippoema/zenodo_record').rstrip('/') + '/'

SIM = DATA_ROOT + 'simulations/'      # simulation inputs and outputs (one folder per set of runs)
AN = DATA_ROOT + 'analysis/'          # results of the analyses, read back by the notebooks
PL = DATA_ROOT + 'plots/'             # figures written by the notebooks
INPUT_STATS = SIM + 'input_stats/'               # force fields and PDB structures of the proteins

# data of the continuum (phase-field) model: phase_field_model/ in the Zenodo record, Le_data/ on the workstation
PHASE_FIELD_DATA = DATA_ROOT + ('phase_field_model/')

REPO = str(Path(__file__).resolve().parents[2]) + '/'    # this repository (notebooks, src/, phase_field/)
STYLES = REPO + 'plot_styles/'                           # matplotlib style files
