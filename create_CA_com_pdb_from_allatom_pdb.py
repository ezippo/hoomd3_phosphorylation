import argparse

import numpy as np
import MDAnalysis as mda

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate pdb file with beads positions for HPS-like models from all-atom pdb.')
    
    parser.add_argument('-i','--infile', required=True, type=str, help='Input all-atom pdb file.')
    
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-ca', '--carbonalpha', action='store_true', help='All the beads will be centered in C_alpha of the residue of the protein.')
    group_mode.add_argument('-cm', '--centerofmass', type=str, help='The beads related to residues of the folded domain will be centered in the center-of-mass of the residue instead of the C_alpha. Argument must be formatted as "x1-y1,x2-y2,...,xn-yn" to specify folded domains from residue x1 to y1, from x2 to y2 etc...')
    
    parser.add_argument('-o','--outfile', default='out.pdb', type=str, help='Output file name.')
    parser.add_argument('-ic','--initialcut', default=0, type=int, help='Remove the first residues up to "initialcut".')
    parser.add_argument('-fc','--finalcut', default=-1, type=int, help='Remove the last residues starting from "finalcut".')

    args = parser.parse_args()

    u = mda.Universe(args.infile)
    ag = u.select_atoms('name CA')

    if not args.carbonalpha:
        folded_domains = args.centerofmass.split(',')
        for domain in folded_domains:
            start_domain, end_domain = np.array(domain.rsplit('-'), dtype=int)

            for res in range(start_domain-1, end_domain):
                ag[res].position = u.residues[res].atoms.center_of_mass()
            print(res)

    ag = ag[args.initialcut:args.finalcut]

    ag.write(args.outfile)
