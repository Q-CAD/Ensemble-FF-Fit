import argparse
import yaml
import os
import sys
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from pymatgen.core.periodic_table import Element, Species
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description="Argument parser to generate substituted structures from POSCAR files")

    parser.add_argument("--poscars_directory", "-pd", help="Path to the directory tree with editable POSCARs", required=True)
    parser.add_argument("--substitution_yaml", "-sy", help="Path to the YAML file with substitution elements and oxidation states, format {'Bi': 3, 'Se': -2}", 
                        required=True)

    # Parse arguments
    args = parser.parse_args()
    generate(args)

    return

def assign_oxidation_states(structure):
    # Use MinimumDistanceNN to guess oxidation states automatically
    try:
        #oxidizer = MinimumDistanceNN()
        #structure.add_oxidation_state_by_guess(oxidizer)
        structure.add_oxidation_state_by_guess()
        return True
    except Exception as e:
        print(f"Failed to assign oxidation states for {structure.composition}: {e}")
        return False

def radius_difference_dct(el_str, oxi_int, sub_dict, match_charge=True):
    rad_diff_dct = {}
    for sub_el_str, sub_oxi_int in sub_dict.items():
        same_sign = np.sign(oxi_int) == np.sign(sub_oxi_int)
        if match_charge == True:
            if same_sign == False:
                continue
            else:
                try:
                    rad_spec, rad_sub_spec = Species(el_str, oxi_int).ionic_radius, Species(sub_el_str, sub_oxi_int).atomic_radius
                    rad_diff_dct[sub_el_str] = np.subtract(rad_spec, rad_sub_spec)
                except TypeError:
                    rad_el, rad_sub_el = Element(el_str).atomic_radius, Element(sub_el_str).atomic_radius
                    rad_diff_dct[sub_el_str] = np.subtract(rad_el, rad_sub_el)
        else:
            try:
                rad_el, rad_sub_el = Element(el_str).atomic_radius, Element(sub_el_str).atomic_radius
                rad_diff_dct[sub_el_str] = np.subtract(rad_el, rad_sub_el)
            except:
                raise TypeError
    return rad_diff_dct

def substitute_species(structure, sub_dict, species_dict):
    # Full radius comparison dictionary
    full_dct = {}
    for el_str, oxi_int in species_dict.items():
        if oxi_int == 0:
            rad_diff_dct = radius_difference_dct(el_str, oxi_int, sub_dict, match_charge=False)
        else:
            rad_diff_dct = radius_difference_dct(el_str, oxi_int, sub_dict, match_charge=True)
        full_dct[el_str] = rad_diff_dct

    # Assign elements based on which are the most similar
    mapping_dct = {}
    for key, sub_dict in full_dct.items():
        # Find the sub-key with the lowest value in the sub-dictionary
        min_sub_items = min(sub_dict.items(), key=lambda x: np.abs(x[1]))
        # Store the sub-key with the lowest value in the result dictionary
        mapping_dct[key] = min_sub_items[0]

    # Can add-in the missing elements to most dissimilar atoms (not built yet)
    to_sub = [el for el in list(sub_dict.keys()) if el not in list(np.unique(list(mapping_dct.values())))]
    print(f'Created substitution dictionary {mapping_dct}; {to_sub} not substituted')

    # Apply the substitution and return the transformed structure
    try:
        transformation = SubstitutionTransformation(mapping_dct)
        new_structure = transformation.apply_transformation(structure)
        return new_structure
    except Exception as e:
        print(f"Substitution failed: {e}")
        return None

def rescale_structure(structure):
    # Rescale the volume of the structure
    try:
        volume_predictor = DLSVolumePredictor()
        rescaled_structure = volume_predictor.get_predicted_structure(structure)
        return rescaled_structure
    except Exception as e:
        print(f"Failed to rescale structure: {e}")
        return None

def process_poscar_files(directory, sub_dict):
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'POSCAR':
                poscar_path = os.path.join(root, file)
                # Load structure from POSCAR
                structure = Structure.from_file(poscar_path)
                original = deepcopy(structure)
                assign_oxidation_states(structure)
                species_dict = {str(specie.element): specie.oxi_state for specie in structure.types_of_specie}

                print(f"Attempting to substitute for structure with species dict {species_dict}")
                # Save the original structure as POSCAR_template
                structure.to(fmt="poscar", filename=os.path.join(root, 'POSCAR_template'))

                # Assign oxidation states
                assign_oxidation_states(structure)
                # Substitute species based on the provided dictionary
                try:
                    substituted_structure = substitute_species(original, sub_dict, species_dict)
                except TypeError:
                    print(f"Error processing file {poscar_path}; skipping file")
                    continue

                # Rescale the structure
                rescaled_structure = rescale_structure(substituted_structure)
                rescaled_structure.to(fmt="poscar", filename=os.path.join(root, 'POSCAR'))
                print(f"Processed and saved modified POSCAR for {poscar_path}")


def generate(args):
    
    with open(args.substitution_yaml, 'r') as file:
        input_dct = yaml.safe_load(file)

    process_poscar_files(args.poscars_directory, input_dct)

    return 

if __name__ == '__main__':
    main()
