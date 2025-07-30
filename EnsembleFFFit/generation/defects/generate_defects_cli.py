#!/usr/bin/env python3

import argparse
from pymatgen.core.structure import Structure
from pymatgen.core import Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.defects.core import Substitution
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.analysis.defects.generators import AntiSiteGenerator
import os
from pathlib import Path
import numpy as np
from copy import deepcopy
from itertools import combinations


def main():
    parser = argparse.ArgumentParser(description="Argument parser to generate vasp_inputs from POSCAR files")
    
    parser.add_argument("--poscars_directory", "-pd", help="Path to the directory tree with editable POSCARs", required=True)
    parser.add_argument("--vacancy", "-v", help="Generate vacancy", action="store_true")
    parser.add_argument("--antisite", "-a", help="Generate antisite", action="store_true")
    parser.add_argument("--interstitial", "-i", help="Generate interstitial", action="store_true")
    parser.add_argument("--substitution", "-s", help="Generate substitution", action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    generate(args)

    return


def _element_str(sp_or_el: Species | Element) -> str:
    """Convert a species or element to a string."""
    if isinstance(sp_or_el, Species):
        return str(sp_or_el.element)
    elif isinstance(sp_or_el, Element):
        return str(sp_or_el)
    else:
        raise ValueError(f"{sp_or_el} is not a species or element") 

def generate_vacancy_set(structure, *args, **kwargs):
    
    """Generate all-possible vacancy defects.
    """
        
    vac_defect = VacancyGenerator()
    struc = {'structures': [] , 'meta-data': []}

    for defect in vac_defect.generate(structure,  *args, **kwargs):
                
        struc['structures'].append(defect.defect_structure)
        struc['meta-data'].append(defect)
                
    return struc

def generate_substitution_set(structure, substitution: dict[str, str | list], **kwargs):
    
    """Generate all-possible substitutional defects.
    """
        
    struc = {'structures': [] , 'meta-data': []}
        
    for defect in custom_generator(structure, substitution, **kwargs):
            
        struc['structures'].append(defect.defect_structure)
        struc['meta-data'].append(defect)
                
    return struc

def generate_antisite_set(structure):
    
    """Generate all-possible anti-site defects.
    """
        
    anti_def = AntiSiteGenerator()
    struc = {'structures': [] , 'meta-data': []}

    for defect in anti_def.generate(structure):
                
        struc['structures'].append(defect.defect_structure)
        struc['meta-data'].append(defect)
                
    return struc

def generate_voronoi_set(structure, insert_species):
    """Generate all-possible voronoi-based interstitial defects.
    """

    int_def = VoronoiInterstitialGenerator()
    struc = {'structures': [] , 'meta-data': []}

    for defect in int_def.generate(structure, insert_species):
                
        struc['structures'].append(defect.defect_structure)
        struc['meta-data'].append(defect)

    return struc

def custom_generator(structure: Structure, substitution: dict[str, str | list], **kwargs):

    " a wrapper function to avoid conflicts in spglib and pymatgen libraries, ref. pyamtgen/analysis/defects"

    sga = SpacegroupAnalyzer(structure)
    sym_struct = sga.get_symmetrized_structure()
    for site_group in sym_struct.equivalent_sites:
        site = site_group[0]
        el_str = _element_str(site.specie)
        if el_str not in substitution.keys():
            continue
        sub_el = substitution[el_str]
        if isinstance(sub_el, str):
            sub_site = PeriodicSite(
                Species(sub_el),
                site.frac_coords,
                structure.lattice,
                properties=site.properties,
            )
            yield Substitution(
                structure,
                sub_site,
                equivalent_sites=[
                    PeriodicSite(
                        Species(sub_el),
                        site.frac_coords,
                        structure.lattice,
                        properties=site.properties,
                    )
                    for site in site_group
                ],
                **kwargs,
            )
        elif isinstance(sub_el, list):
            for el in sub_el:
                sub_site = PeriodicSite(
                    Species(el),
                    site.frac_coords,
                    structure.lattice,
                    properties=site.properties,
                )
                yield Substitution(
                    structure,
                    sub_site,
                    equivalent_sites=[
                        PeriodicSite(
                            Species(el),
                            site.frac_coords,
                            structure.lattice,
                            properties=site.properties,
                        )
                        for site in site_group
                    ],
                    **kwargs)

def write_text(text, path):
    with open(path, 'w') as write_file:
        write_file.write(text)
    return

def write_data(data, directory, defect):
    os.mkdir(directory)
    for i, structure in enumerate(data['structures']):
        # To prepare for VASP submission
        structure.remove_oxidation_states()
        sorted_structure = structure.get_sorted_structure()

        subdirectory = os.path.join(directory, f'{defect}{i}')
        os.mkdir(subdirectory)
        filename = os.path.join(subdirectory, 'POSCAR')
        sorted_structure.to(filename, fmt='poscar')
        metadata_path = os.path.join(subdirectory, 'METADATA')
        write_text(str(data['meta-data'][i]) + '\n', metadata_path)
    return 

def generate(args):
    top_level = Path(args.poscars_directory).absolute()
    abs_poscars_directory = os.path.abspath(args.poscars_directory)

    for root, dirs, files in os.walk(abs_poscars_directory):
        possible_poscar = os.path.join(root, 'POSCAR')
        if os.path.exists(possible_poscar):
            structure = Structure.from_file(possible_poscar)
            
            # Generate the vacancy structures
            if args.vacancy:
                try:
                    vacancy_dir = os.path.join(root, f'vacancy')
                    vacancy_data = generate_vacancy_set(deepcopy(structure))
                    write_data(vacancy_data, vacancy_dir, 'vacancy')
                except:
                    print(f'Vacancies not generated for {root}')

            # Generate the antisite structures
            if args.antisite:
                try:
                    antisite_dir = os.path.join(root, f'antisite')
                    antisite_data = generate_antisite_set(deepcopy(structure))
                    write_data(antisite_data, antisite_dir, 'antisite')
                except:
                    print(f'Anti-site defects not generated for {root}')
            
            # Generate the interstitial data
            if args.interstitial:
                els = [str(el) for el in structure.elements]
                try:
                    interstitial_dir = os.path.join(root, f'interstitial')
                    interstitial_data = generate_voronoi_set(deepcopy(structure), els)
                    write_data(interstitial_data, interstitial_dir, 'interstitial')
                except:
                    print(f'Interstitial data not generated for {root}')

            # Generate the substitution data
            if args.substitution:
                unique_els = list(np.unique([str(el) for el in structure.elements]))
                sub_dct = {x: [y for y in unique_els if y != x] for x in unique_els}
                print(sub_dct)
                try:
                    substitution_dir = os.path.join(root, f'substitution')
                    substitution_data = generate_substitution_set(deepcopy(structure), sub_dct)
                    write_data(substitution_data, substitution_dir, 'substitution')
                except NameError:
                    print(f'Substitution data not generated for {root}')
            
            os.rename(possible_poscar, os.path.join(root, 'POSCAR.vasp'))

if __name__ == '__main__':
    main()

