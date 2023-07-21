import numpy as np
import argparse

import hps_phosphorylation.hoomd_util as hu
import hps_phosphorylation.phosphorylation as phospho
import hps_phosphorylation.hps_like_models as hps

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MD simulation in HOOMD3 using HPS-like models')
    