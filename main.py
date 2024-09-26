# -*- coding: utf-8 -*-

import argparse

import hps_phosphorylation.hoomd_util as hu
import hps_phosphorylation.hps_like_models as hps

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MD simulation in HOOMD3 using HPS-like models')
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-c', '--create_conf', action='store_true', help='The code will run in the create_initial_configuration mode. Only cubic box are available for creating the initial configuration, the box can be resized during the simulation using the flag -br. Give only one number in the input "box". ')
    group_mode.add_argument('-m', '--model', type=str, choices=['HPS', 'HPS_cp', 'CALVADOS2'], help='The code will run in simulation mode. The argument of this flag must be the name of the coarse-grained model to use in the simulation.')
    parser.add_argument('-i','--infile', required=True, type=str, help='Input file with simulation parameters, logging file name and parameters, system file name.')
    parser.add_argument('-r', '--rescale', default=0, type=float, help='Scale down rigid body interaction by X percentage. To use also in create_initial_configuration mode to incude the rescaled rigid body types (value of argmuent not important in this case).')
    parser.add_argument('-br', '--boxresize', default=None, nargs=3, type=float, help='The simulation will be used to resize the box from the initial configuration to the sizes given in the argument. The argument should be a list with the side lengths (Lx, Ly, Lz).')
    parser.add_argument('-d', '--displactivesite', default=None, type=str, help='To use in case of displacement of active site compared to the positions of the beads. Give as argument the path to the file with xyz coordinates of the displaced active site in Angstrom. Displacement compared to the positions in the pdb given in the infile. ')

    parser.add_argument('--mode', default='relax', type=str, choices=['relax', 'ness', 'nophospho'], help='Default phosphorylation is active without exchange SER/SEP with the chemical bath. If ness also exchange step is added. If nophospho phosphorzlation and exchange are deactivated.' )
    
    args = parser.parse_args()

    ## READ INPUT FILE
    macro_dict = hu.macros_from_infile(args.infile)
    print(args.infile)
    aa_param_dict = hu.aa_stats_from_file(macro_dict['stat_file'])
    syslist = hu.system_from_file(macro_dict['sysfile'])

    # create_initial_configuration mode
    if args.create_conf:
        # only cubic box are available for creating the initial configuration, the box can be resized during the simulation using the flag -br. Give only one number in the input "box".
        box_length = float( macro_dict['box'] )
        hps.create_init_configuration(filename=macro_dict['logfile']+'_start.gsd', syslist=syslist, aa_param_dict=aa_param_dict, 
                                      box_length=box_length, rescale=bool(args.rescale)) 
    # simulation mode
    else:
        hps.simulate_hps_like(macro_dict=macro_dict, aa_param_dict=aa_param_dict, syslist=syslist, model=args.model, 
                              rescale=args.rescale, mode=args.mode, resize=args.boxresize, displ_active_site=args.displactivesite)
        