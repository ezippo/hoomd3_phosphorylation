import numpy as np

import hps_phosphorylation.hoomd_util as hu
import hps_phosphorylation.phosphorylation as phospho
import hps_phosphorylation.hps_like_models as hps

if __name__=='__main__':
    infile = '../input0.in'
    hps.simulate_hps_like(infile, model='HPS', rescale=0)