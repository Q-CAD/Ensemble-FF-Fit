import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ase.io import read
from multiprocessing import Pool, cpu_count
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN

def main():
    parser = argparse.ArgumentParser(description="Argument parser to run LAMMPs with Flux using Python")
    
    # Run and input directories
    parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
    parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')

    # Input files
    parser.add_argument("--check_file", "-cf", help="Name of LAMMPs data file to check for in the --run_directory", default='data.npt_relax')
    parser.add_argument("--structure", "-s", type=str, help="Name of the .lmp file", default='structure.lmp') # Build out for continuation jobs here
    parser.add_argument("--json_file", "-jf", type=str, help="Name of the .json file", default='comparison.json')

    # Read file options
    parser.add_argument("--atom_style", "-as", help="LAMMPs structure file atom style", type=str, default='charge')
    parser.add_argument("--oxi_dct", "-od", help="Pymatgen oxidation dictionary in .json format, e.g., '{\"Bi\":3,\"Se\":-2}'", type=json.loads)
    parser.add_argument("--use_weights", "-uw", help="Use weights for pymatgen's CN analysis", type=bool, default=True)

    args = parser.parse_args()
    cn_check(args)

def compute_cn_diff(use_args):
    """ Helper function to compute the coordination number difference. """
    cnn, ref_cn_dct, p_tup, p_tup_dct, use_weights  = use_args

    site_inds = [i for i in range(len(p_tup_dct['structure']))]
    site_els = [str(p_tup_dct['structure'][i].specie.element) for i in range(len(p_tup_dct['structure']))] 
    unique_site_els = list(np.unique(site_els))
    unique_site_dct = {el: None for el in unique_site_els}

    # Assume that the site elements in the reference and the 
    for uel in unique_site_els:
        matched_els_is = [i for i in range(len((p_tup_dct['structure']))) if str(p_tup_dct['structure'][i].specie.element) == uel]
        matched_els_cns = [cnn.get_cn(p_tup_dct['structure'], i, use_weights=use_weights) for i in matched_els_is]
        ref_els_cns = [ref_cn_dct[p_tup_dct['name']][i] for i in matched_els_is]
        norm_diff = np.linalg.norm(np.subtract(matched_els_cns, ref_els_cns))
        unique_site_dct[uel] = norm_diff

    return (p_tup, unique_site_dct)

def get_average_coordination_deviation(args, ref_path_dct, path_dictionary):
    print('Constructing reference coordination numbers...')
    cnn = CrystalNN(weighted_cn=args.use_weights)
    ref_cn_dct = {}
    for key in list(ref_path_dct.keys()): # Name of the MD
        structure = ref_path_dct[key]
        ref_site_cns = []
        for site_ind in range(len(structure)):
            cn = cnn.get_cn(structure, site_ind, use_weights=args.use_weights)
            ref_site_cns.append(cn)
        ref_cn_dct[key] = ref_site_cns

    args_list = [
        (cnn, ref_cn_dct, p_tup, p_tup_dct, args.use_weights)
        for p_tup, p_tup_dct in path_dictionary.items()
    ]

    print('Performing multiprocessing analysis...')
    # Use multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_cn_diff, args_list), total=len(args_list)))

    # Collect results
    structure_dct = {}
    for p_tup, norm_dct in results:
        key_name = str(Path(p_tup).parent) # Path to the force field
        subdict_key = str(Path(p_tup).name) # Name of path to the MD run
        if key_name not in structure_dct.keys():
            structure_dct[key_name] = {}
        structure_dct[key_name][subdict_key] = norm_dct

    return structure_dct

def comparison_paths(args):
    print('Building reference dictionary...')
    ref_dct = {}
    for root, _, _ in os.walk(args.inputs_directory):
        structure_path = os.path.join(root, args.structure)
        if os.path.exists(structure_path):
            md_name = Path(structure_path).parent.name
            ld = LammpsData.from_file(structure_path, atom_style=args.atom_style)
            oxi_ref = ld.structure.add_oxidation_state_by_element(args.oxi_dct)
            ref_dct[md_name] = oxi_ref

    print('Finding comparison paths...') 
    path_dictionary = {}
    aaa = AseAtomsAdaptor()
    for root, _, _ in os.walk(args.run_directory):
        check_structure_path = os.path.join(root, args.check_file)
        if os.path.exists(check_structure_path):
            parent_name = Path(check_structure_path).parent.name
            ld = LammpsData.from_file(check_structure_path, atom_style=args.atom_style, sort_id=True)
            oxi_s = ld.structure.add_oxidation_state_by_element(args.oxi_dct)
            path_dictionary[root] = {'name': parent_name, 
                                     'structure': oxi_s} 

    return ref_dct, path_dictionary

def cn_check(args):
    ref_dct, structure_dct = comparison_paths(args)
    deviation_dct = get_average_coordination_deviation(args, ref_dct, structure_dct)
    
    with open(args.json_file, 'w') as json_file:
        json.dump(deviation_dct, json_file, indent=4)
    return 

if __name__ == '__main__':
    main()

