# -*- coding: utf-8 -*-

import numpy as np
import argparse

import hps_phosphorylation.hoomd_util as hu
#import hps_phosphorylation.phosphorylation as phospho
#import hps_phosphorylation.hps_like_models as hps

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MD simulation in HOOMD3 using HPS-like models')
    parser.add_argument('-i','--infile', required=True, type=str, help='Input file with simulation parameters, logging file name and parameters, system file name')
    parser.add_argument('-m', '--model', required=True, type=str, choices=['HPS', 'HPS_cp', 'CALVADOS2'], help='Coarse-grained model to use in the simulation.')
    parser.add_argument('-r', '--rescale', type=float, default=0, help='Scale down rigid body interaction by X%')

    args = parser.parse_args()
    print(args.infile)  