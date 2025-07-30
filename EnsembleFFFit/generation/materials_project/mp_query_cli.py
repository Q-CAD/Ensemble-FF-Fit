import argparse
import yaml
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import sys

CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'api_key.yml')

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

def main():
    config_dct = load_config()

    parser = argparse.ArgumentParser(description="Argument parser to generate vasp_inputs from POSCAR files")

    parser.add_argument("--poscars_directory", "-pd", type=str, help="Path to the directory where POSCARS will be written", required=True)
    parser.add_argument("--max_atoms", "-ma", type=int, help="Maximium number of atoms for structures", default=0)
    parser.add_argument("--MP_yaml", "-my", type=str, help="Path to a .yml file with MPIds or compositions inputs", required=True)
    parser.add_argument("--MP_api_key", "-mpapi", type=str, help="API key for Materials Project queries", default=config_dct.get('MP_API_KEY', ''))
    parser.add_argument("--field_order", "-fo", nargs='+', help="Structure of written directory", default=['chemsys', 'formula_pretty', 'material_id']) 
    
    # Parse arguments
    args = parser.parse_args()
    query_mp(args)

    return

def get_conventional(structure):
    sga = SpacegroupAnalyzer(structure)
    conventional_structure = sga.get_conventional_standard_structure()
    return conventional_structure

def pull_from_MP(api_key, yml, ymlkey, fields):
    with MPRester(api_key) as mpr:
        if ymlkey == 'mpids':
            docs = mpr.summary.search(material_ids=yml[ymlkey], fields=fields)
        elif ymlkey == 'chemsys':
            docs = mpr.summary.search(chemsys=yml[ymlkey], fields=fields)
        else:
            print(f'{field} not supported!')
            sys.exit(1)
    return docs

def MPQuery(api_key, yml, ymlkey, fields):
    try: # Attempt to pull from material ids
        docs = pull_from_MP(api_key, yml, ymlkey, fields)
    except KeyError:
        docs = None
    return docs 

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as load_file:
        yaml_data = yaml.safe_load(load_file)
    return yaml_data

def check_dir_exists(dirpath):
    if os.path.exists(dirpath):
        pass
    else:
        os.mkdir(dirpath)
    return

def write_data(docs, path, order, max_atoms, conventional=True):
    for doc in docs:
        start_path = path
        if conventional == True:
            structure = get_conventional(doc.structure).remove_oxidation_states()
        else:
            structure = docs.structure.remove_oxidation_states()

        if max_atoms:
            if len(structure) > max_atoms:
                print(f'{structure.formula} exceeds {max_atoms} atoms; skipping')
                continue

        for o in order: # Build directory tree
            check_path = os.path.join(start_path, getattr(doc, o))
            check_dir_exists(check_path)
            start_path = check_path
        write_path = os.path.join(start_path, 'POSCAR')

        structure.to(write_path, fmt='poscar')
    return 

def query_mp(args):
    abs_path = os.path.abspath(args.poscars_directory)
    check_dir_exists(abs_path)
    fields = ['structure'] + args.field_order

    input_yaml = load_yaml(args.MP_yaml)
    mdocs = MPQuery(args.MP_api_key, input_yaml, 'mpids', fields)
    cdocs = MPQuery(args.MP_api_key, input_yaml, 'chemsys', fields)

    if mdocs is not None:
        write_data(mdocs, abs_path, args.field_order, args.max_atoms)
    if cdocs is not None:
        write_data(cdocs, abs_path, args.field_order, args.max_atoms)
    return

if __name__ == '__main__':
    main()

