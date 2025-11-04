import os
import shutil
import re
import sys
import argparse
import warnings
from pathlib import Path
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile
from pymatgen.core.periodic_table import Element

def main():
    parser = argparse.ArgumentParser(description='Format output files to an new directory')

    parser.add_argument("--source_directory", "-sd", help="Path to the source directory tree", default='outputs')
    parser.add_argument("--num_source_directory_directories", "-nsdd", help="Number of source directory directories (from base) to include when writing", 
                         type=int, default=3)
    parser.add_argument("--target_directory", "-td", help="Path to the target directory", default='ffields')
    parser.add_argument("--target_name", "-tn", help="Name of target file", default='ffield')
    parser.add_argument("--pattern", "-p", help="Patterns, e.g., '^new_FF_([\d_]+)$', '^dump_(\d+)\.dump$', '^MACE_MatEnsemble_stage([^.]+)\\.model$', '^MACE_([^._-]+)\\.model$'", 
                        default='^new_FF_([\d_]+)$')
    parser.add_argument("--in_lammps", "-in", help="Exclusively used by dump files; path to in.lammps file for atom mapping") 
    parser.add_argument("--atom_style", "-at", help="LAMMPs structure file atom style, e.g., 'full', 'charge', 'atomic'", 
                        default='charge', choices=['full', 'charge', 'atomic']) 
    
    # JaxReaxFF example: '^new_FF_([\d_]+)$'
    # LAMMPs example: '^dump_(\d+)\.dump$'
    # MACE example: '^MACE_MatEnsemble_stage([^.]+)\\.model$' or '^MACE_([^._-]+)\\.model$'
    
    args = parser.parse_args()
    copy_and_rename_files(args)
    return 

def get_atom_mapping_from_control(path):
    lif = LammpsInputFile.from_file(path)
    pair_coeff_lists = [line for line in lif.as_dict()['stages'][0]['commands'] if 'pair_coeff' in line]
    
    elements = []
    for pair_coeff_list in pair_coeff_lists:
        for split in pair_coeff_list[-1].split():
            try:
                el = Element(split)
                elements.append(str(el))
            except:
                pass 
    mapping = {str(Element.from_Z(i+1)): str(elements[i]) for i in range(len(elements))}

    return mapping

def get_LammpsData_from_dump(path, mapping, atom_style):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = read(path, format='lammps-dump-text')
        structure = AseAtomsAdaptor().get_structure(atoms)
        structure = structure.replace_species(mapping)
        ld = LammpsData.from_structure(structure, atom_style=atom_style)
        return ld

def copy_and_rename_files(args):
    # Ensure the source directory exists
    if not os.path.exists(args.source_directory):
        print(f"Source directory '{args.source_directory}' does not exist.")
        return

    # Ensure the target directory exists, if not, create it
    if not os.path.exists(args.target_directory):
        os.makedirs(args.target_directory)
        print(f"Created target directory '{args.target_directory}'.")

    file_pattern = re.compile(args.pattern)

    # Iterate through files in the source directory
    for root, dirs, files in os.walk(args.source_directory):
        for f in files:
            source_file = os.path.join(root, f)
            if os.path.isfile(source_file) and file_pattern.match(f):
                print(source_file)
                # Extract the unique identifier from the file name
                file_match = file_pattern.match(f)
                if file_match:
                    unique_file_name = file_match.group(1)

                    # Create a subdirectory in the target directory with the unique name
                    #destination_subdir = os.path.join(replace_first_dir(root, args.target_directory), unique_file_name)
                    number = args.num_source_directory_directories
                    middle_directories = Path(root).parts[-number:]
                    destination_subdir = os.path.join(args.target_directory, *middle_directories, unique_file_name)
                    if not os.path.exists(destination_subdir):
                        os.makedirs(destination_subdir)

                    # Copy the file and rename it to "ffield" in the new directory
                    destination_file = os.path.join(destination_subdir, args.target_name)
                    if 'dump' in source_file:
                        mapping = get_atom_mapping_from_control(args.in_lammps)
                        ld = get_LammpsData_from_dump(source_file, mapping, args.atom_style)
                        try:
                            ld.write_file(filename=destination_file)
                            print(f"Converted '{source_file}' dump file to lammps file at '{destination_file}'")
                        except Exception as e:
                            print(f"Failed to convert '{source_file}' dump file and write")
                    else:
                        try:
                            shutil.copy2(source_file, destination_file)
                            print(f"Copied '{source_file}' to '{destination_file}'")
                        except Exception as e:
                            print(f"Failed to copy '{source_file}' to '{destination_file}': {e}")

if __name__ == "__main__":
    main()

