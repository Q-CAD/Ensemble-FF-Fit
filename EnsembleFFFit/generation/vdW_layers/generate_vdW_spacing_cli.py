#!/usr/bin/env python3

import argparse
from pymatgen.core.structure import Structure
from vdW_structures.vdW_structure import VdWStructure
import os
import numpy as np
from pathlib import Path
from copy import deepcopy
import math

def main():
    parser = argparse.ArgumentParser(description="Argument parser to generate vasp_inputs from POSCAR files")
    parser.add_argument("--poscars_directory", "-pd", help="Path to the directory tree with editable POSCARs", required=True)
    parser.add_argument("--displacement_spacing", "-ds", type=float, help="Spacing for the displacement of vdW layers", default=0.03)
    parser.add_argument("--total_displacements", "-td", type=int, help="Number of displacements to run calculations for", default=11)
    parser.add_argument("--displace_layer_indices", "-dli", nargs='+', type=int, help="Indices corresponding to the layers to displace", default=[]) 
    parser.add_argument("--layer_tolerance", "-lt", type=float, help="Tolerance to determine a vdW layer", required=True)
    parser.add_argument("--dry_run", "-dry", action='store_true', help="Say what would be generated")
    # Parse arguments
    args = parser.parse_args()
    displace_layers(args)    

    return args

def spacing_factor(args):
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

def rescale_spacing(vdw_structure, layer_indice, spacing_fraction, args):
    shift_to = vdw_structure.vdW_spacings[layer_indice] * spacing_fraction
    new_vdw_structure = vdw_structure.z_solve_shift_vdW_layers(layer_inds=layer_indice,
                                                               solve_shift=shift_to) 
    possible_min_gaps = [shift_to] + [args.layer_tolerance] + new_vdw_structure.vdW_spacings
    new_minimum = min(possible_min_gaps) - 1e-3 # Get the newest minimum vdW spacing

    newest_vdw_structure = VdWStructure(new_vdw_structure.structure, minimum_vdW_gap=new_minimum)
    layer_ind = is_close_idx(shift_to, newest_vdw_structure.vdW_spacings)
    return newest_vdw_structure, layer_ind

def is_close_idx(target, lst):
    idx = next(
            (i for i, v in enumerate(lst) if math.isclose(v, target, rel_tol=1e-9, abs_tol=1e-12)), None)
    return idx

def displace_layers(args):
    top_level = Path(args.poscars_directory).absolute()
    abs_poscars_directory = os.path.abspath(args.poscars_directory)
    spacing_fractions = spacing_factor(args)

    use_roots, use_poscars = [], []
    for root, dirs, files in os.walk(abs_poscars_directory):
        possible_poscar = os.path.join(root, 'POSCAR')
        possible_vasprun = os.path.join(root, 'vasprun.xml')
        if os.path.exists(possible_poscar) is True and os.path.exists(possible_vasprun) is False: # Don't write in directories with runs
            use_roots.append(root)
            use_poscars.append(possible_poscar)

    for i, root in enumerate(use_roots):
        structure = Structure.from_file(use_poscars[i])
        vdw_structure = VdWStructure(structure, minimum_vdW_gap=args.layer_tolerance)
        for layer_indice in args.displace_layer_indices:
            layer_copy = deepcopy(vdw_structure)
            for sf in spacing_fractions:
                vdw_structure_copy = deepcopy(layer_copy)
                scaled_vdW, layer_ind = rescale_spacing(vdw_structure_copy, layer_indice, sf, args)
                spacing_dir = os.path.join(root, f'{layer_indice}', f'spacing_{sf}')
                scaled_path = os.path.join(spacing_dir, 'POSCAR')
                print(f'POSCAR with layer indice {layer_indice} and spacing {vdw_structure.vdW_spacings[layer_indice]} rescaled to {scaled_vdW.vdW_spacings[layer_ind]} written to {scaled_path}')
                if not args.dry_run:
                    os.makedirs(spacing_dir, exist_ok=True)
                    scaled_vdW.structure.to(scaled_path, fmt='poscar')
        if not args.dry_run:
            os.rename(use_poscars[i], os.path.join(root, 'POSCAR.vasp'))

    return 

if __name__ == '__main__':
    main()

