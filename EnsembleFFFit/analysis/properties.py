import os
import re
import glob
from pathlib import Path
from parse2fit.tools.unitconverter import UnitConverter
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.io.lammps.outputs import parse_lammps_log
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from EnsembleFFFit.utils.copy_by_pattern_cli import get_atom_mapping_from_control
import numpy as np
from copy import deepcopy

def nested_set(dct, keys, value):
    """
    In dct, create the nested path given by keys[0], keys[1], …, keys[-1],
    then assign dct[keys[0]][keys[1]]…[keys[-1]] = value.
    """
    if len(keys) == 1:
        # Base case: one key left → assign the value
        dct[keys[0]] = value
    else:
        # If the first key doesn't exist or isn't a dict, overwrite with a dict
        if keys[0] not in dct or not isinstance(dct[keys[0]], dict):
            dct[keys[0]] = {}
        # Recurse into the next level
        nested_set(dct[keys[0]], keys[1:], value)

def get_energy(log_path, image, energy_label, units):
    log = parse_lammps_log(log_path)
    energy = None
    for l in log: 
        try:
            energy = float(l.loc[l["Step"] == image, energy_label].iloc[0])  # No error message written
        except IndexError:
            try:
                energy = float(l.loc[l["Step"] == str(image), energy_label].iloc[0])  # Error message written
            except:
                continue
    
    uc = UnitConverter()
    if energy:
        if units == 'metal': # Compatible with VASP DFT
            pass
        elif units == 'real': # In kcal/mol
            energy = uc.convert(energy, 'kcal/mol', 'eV/atom', 'energy')
    return energy

def get_atoms(dump_path, mapping):
    try:
        atoms_we = read(dump_path, index=-1, format='lammps-dump-text')
    except StopIteration:
        print(f'Cannot parse {dump_path}')
        return None
    new_symbols = [mapping[sym] for sym in atoms_we.get_chemical_symbols()]
    atoms_orig = deepcopy(atoms_we)
    atoms_we.set_chemical_symbols(new_symbols)
    atoms_we.arrays['forces'] = atoms_orig.get_forces()
    return atoms_we

def get_forces(dump_path, units):
    final_image = next(parse_lammps_dumps(dump_path))
    index = final_image.data['id']
    fxs = final_image.data['fx']
    fys = final_image.data['fy']
    fzs = final_image.data['fz']
    zipped_list = list(zip(index, fxs, fys, fzs))
    sorted_zipped_list = sorted(zipped_list)
    index, fxs, fys, fzs = list(zip(*sorted_zipped_list))
    uc = UnitConverter()
    if units == 'metal':
        pass
    elif units == 'real':
        fxs = [uc.convert(fx, 'kcal/mol', 'eV/atom', 'energy') for fx in fxs]
        fys = [uc.convert(fy, 'kcal/mol', 'eV/atom', 'energy') for fy in fys]
        fzs = [uc.convert(fz, 'kcal/mol', 'eV/atom', 'energy') for fz in fzs]
    return fxs, fys, fzs

def parse_single_points(path_to_images,
                            dump_index = 0,
                            energy_label='PotEng',
                            units='metal',
                        ffield_label=(-5, -3)):
    data = {}
    for root, _, _ in os.walk(os.path.abspath(path_to_images)):
        log_paths = glob.glob(os.path.join(root, '*.lammps'))
        for log_path in log_paths:
            
            p = Path(log_path)
            try:
                element_mapping = get_atom_mapping_from_control(p)
            except ValueError:
                print(f'Cannot parse elements from {p}')
                continue
            dump_paths = glob.glob(os.path.join(root, "*.dump"))
            sorted_dump_path_names = sorted([Path(p).name for p in dump_paths], key=lambda x: int(x.split('_')[1].split('.')[0]))
            index_path = os.path.join(root, sorted_dump_path_names[dump_index])
            image = int(re.findall(r'\d+', sorted_dump_path_names[dump_index])[0])
            atoms = get_atoms(index_path, element_mapping)
            
            try:
                energy = get_energy(log_path, image, energy_label, units=units)
            except:
                energy = np.nan
            
            try:
                fxs, fys, fzs = get_forces(os.path.join(root, index_path), units=units)
            except Exception:
                fxs, fys, fzs = [], [], []
            
            if atoms:
                atoms.info['energy'] = energy
                md = p.parent.parent.name
                original_image = int(re.findall(r'\d+', p.parent.name)[0])
                ffield_parts = root.split("/")
                ffield = "_".join(ffield_parts[ffield_label[0]:ffield_label[1]])
                nested_set(data, [ffield, md, original_image], {'energy': energy, 'atoms': atoms,
                                                   'fx': fxs, 'fy': fys, 'fz': fzs})
    return data

def parse_VASP_single_points(path_to_runs):
    data = {}
    for root, _, _ in os.walk(path_to_runs):
        vasprun_path = os.path.join(root, 'vasprun.xml')
        if os.path.exists(vasprun_path):
            v = Vasprun(vasprun_path)
            energy = v.final_energy
            len_structure = len(v.ionic_steps[-1]['forces'])
            fxs = [v.ionic_steps[-1]['forces'][i][0] for i in range(len_structure)]
            fys = [v.ionic_steps[-1]['forces'][i][1] for i in range(len_structure)]
            fzs = [v.ionic_steps[-1]['forces'][i][2] for i in range(len_structure)]

            p = Path(vasprun_path)
            image = int(p.parent.name) #int(re.findall(r'\d+', p.name)[0])
            md = p.parent.parent.name
            ffield = p.parent.parent.parent.name
            nested_set(data, [ffield, md, image], {'energy': energy, 'structure': v.final_structure, 
                                                   'fx': fxs, 'fy': fys, 'fz': fzs})
    return data

def comparison_dictionary(data_dictionary, ref_key='DFT', rel_image=None):
    def safe_subtract(a, b):
        try:
            return np.subtract(a, b)
        except Exception:
            return None

    dev_data = {}
    for ffield in list(data_dictionary.keys()):
        for md in list(data_dictionary[ffield].keys()):
            for image in list(data_dictionary[ffield][md].keys()):
                
                # Get energy difference first
                energy = data_dictionary[ffield][md][image]['energy']
                ref_energy = data_dictionary[ref_key][md][image]['energy']
                if rel_image is not None:
                    use_energy = safe_subtract(energy, data_dictionary[ffield][md][rel_image]['energy'])
                    use_ref = safe_subtract(ref_energy, data_dictionary[ref_key][md][rel_image]['energy']) 
                else:
                    use_energy = energy
                    use_ref = ref_energy
                energy_difference = safe_subtract(use_ref, use_energy)
                energy_difference = np.abs(energy_difference) if energy_difference is not None else None

                # Get force differences next
                ref_fx = data_dictionary[ref_key][md][image]['fx']
                fx_difference = safe_subtract(ref_fx, data_dictionary[ffield][md][image]['fx'])
                fx_rmsd = np.sqrt(np.mean(fx_difference ** 2)) if fx_difference is not None else None
                ref_fy = data_dictionary[ref_key][md][image]['fy']
                fy_difference = safe_subtract(ref_fy, data_dictionary[ffield][md][image]['fy'])
                fy_rmsd = np.sqrt(np.mean(fy_difference ** 2)) if fy_difference is not None else None
                ref_fz = data_dictionary[ref_key][md][image]['fz']
                fz_difference = safe_subtract(ref_fz, data_dictionary[ffield][md][image]['fz'])
                fz_rmsd = np.sqrt(np.mean(fz_difference ** 2)) if fz_difference is not None else None

                # Get the dictionary
                nested_set(dev_data, [ffield, md, image], {'energy': energy_difference,
                                                       'fx': fx_rmsd, 
                                                       'fy': fy_rmsd, 
                                                       'fz': fz_rmsd})

    return dev_data

def performance_rank(deviation_data, md, energy_weight=1, force_weight=1):
    md_dictionary = {}
    for ff in list(deviation_data.keys()):
        md_dictionary[ff] = {}
        energy = 0
        force = 0
        images = list(deviation_data[ff][md].keys())
        for image in images:
            energy += deviation_data[ff][md][image]['energy']
            force += deviation_data[ff][md][image]['fx']
            force += deviation_data[ff][md][image]['fy']
            force += deviation_data[ff][md][image]['fz']
        energy = energy / len(images)
        force = force / (len(images) * 3)
        md_dictionary[ff]['energy'] = energy
        md_dictionary[ff]['force'] = force
        md_dictionary[ff]['score'] = (energy_weight * energy) + (force_weight * force)
    sorted_items = sorted(md_dictionary.items(), key=lambda kv: kv[1]['score'])
    print(f"MD Image: {md}")
    for key, inner in sorted_items:
        print(f"{key}: energy = {inner['energy']}, force = {inner['force']}, score = {inner['score']}")
    return md_dictionary, sorted_items

def write_poscars(data_dictionary, to_path, filename='POSCAR'):
    aaa = AseAtomsAdaptor()
    for top_level, tl_value in data_dictionary.items():
        for mid_level, ml_value in tl_value.items():
            for bottom_level, bl_value in ml_value.items():
                atoms = bl_value['atoms']
                structure = aaa.get_structure(atoms).sort() 
                dir_path = os.path.join(str(to_path), str(top_level), str(mid_level), str(bottom_level))
                os.makedirs(dir_path, exist_ok=True)
                structure.to(os.path.join(dir_path, 'POSCAR'))

    return 


