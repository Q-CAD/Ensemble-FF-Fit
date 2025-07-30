#!/usr/bin/env python3

import argparse
from pymatgen.core.structure import Structure
import os
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Argument parser to generate vasp_inputs from POSCAR files")
    parser.add_argument("--poscars_directory", "-pd", help="Path to the directory tree with editable POSCARs", required=True)
    parser.add_argument("--displacement_spacing", "-ds", type=float, help="Spacing for the displacement of structures volumes", default=0.03)
    parser.add_argument("--total_displacements", "-td", type=int, help="Number of displacements to run calculations for", default=11)
    
    # Parse arguments
    args = parser.parse_args()
    EoS(args)    

    return args

def volume_factor(args):
    odd = False
    if args.total_displacements % 2 == 1: # Odd total number
        odd = True

    if odd == True: # Centered at 0 displacement
        positive_lst = [(i+1)*args.displacement_spacing for i in range(int((args.total_displacements-1)/2))]
    else: # Even total number, off-centered
        positive_lst = [(i+1)*(args.displacement_spacing/2) for i in range(int(args.total_displacements/2))]

    if odd == True:
        return [1-p for p in positive_lst] + [1] + [1+p for p in positive_lst]
    else:
        return [1-p for p in positive_lst] + [1+p for p in positive_lst]

def rescale_volume(structure, volume_fraction):
    scaling = volume_fraction ** (1/3)
    new_matrix = structure.lattice.matrix * scaling
    new_structure = Structure(lattice=new_matrix, species=structure.species, coords=structure.frac_coords, 
                              coords_are_cartesian=False)
    return new_structure

def EoS(args):
    top_level = Path(args.poscars_directory).absolute()
    abs_poscars_directory = os.path.abspath(args.poscars_directory)
    volume_fractions = volume_factor(args)

    for root, dirs, files in os.walk(abs_poscars_directory):
        possible_poscar = os.path.join(root, 'POSCAR')
        possible_vasprun = os.path.join(root, 'vasprun.xml')
        if os.path.exists(possible_poscar) is True and os.path.exists(possible_vasprun) is False: # Don't write in directories with runs
            structure = Structure.from_file(possible_poscar)
            for vf in volume_fractions:
                volume_dir = os.path.join(root, f'volume_{vf}')
                try:
                    os.mkdir(volume_dir)
                except OSError as e: # Already exists
                    pass
                scaled = rescale_volume(structure, vf)
                scaled_path = os.path.join(volume_dir, 'POSCAR')
                scaled.to(scaled_path, fmt='poscar')
                svolume_factor = np.round(scaled.lattice.volume / structure.lattice.volume, 5)
                print(f'POSCAR with volume factor {svolume_factor} written to {scaled_path}')
            os.rename(possible_poscar, os.path.join(root, 'POSCAR.vasp'))

if __name__ == '__main__':
    main()

