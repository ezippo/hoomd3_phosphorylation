# -*- coding: utf-8 -*-

import numpy as np
import argparse

import hps_phosphorylation.hoomd_util as hu
#import hps_phosphorylation.phosphorylation as phospho
#import hps_phosphorylation.hps_like_models as hps

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MD simulation in HOOMD3 using HPS-like models')
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-c', '--create_conf', type=str, help='The code will run in the create_initial_configuration mode. The argument of this flag must be the name of the output configuration file. Give sysfile.dat as inputfile.')
    parser.add_argument('-m', '--model', required=True, type=str, choices=['HPS', 'HPS_cp', 'CALVADOS2'], help='The code will run in simulation mode. The argument of this flag must be the name of the coarse-grained model to use in the simulation.')
    parser.add_argument('-i','--infile', required=True, type=str, help='In mode simulation: input file with simulation parameters, logging file name and parameters, system file name. In mode create_initial_configuration: sysfile.dat .')
    parser.add_argument('-r', '--rescale', type=float, help='Scale down rigid body interaction by X% . To use also in create_initial_configuration mode to incude the rescaled rigid body types (value of argmuent not important in this case).')

    args = parser.parse_args()
    print(args.infile)  
    