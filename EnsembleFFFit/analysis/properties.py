import os
import re
import glob
from pathlib import Path
from parse2fit.tools.unitconverter import UnitConverter
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.io.lammps.outputs import parse_lammps_log
from pymatgen.io.vasp.outputs import Vasprun
import numpy as np

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

def get_energy(log_path, log_index, energy_label, units):
    log = parse_lammps_log(log_path)
    try:
        energy = float(log[log_index][energy_label][1])  # Error message written
    except:
        energy = float(log[log_index][energy_label][0])  # No error message written
    uc = UnitConverter()
    if units == 'metal': # Compatible with VASP DFT
        pass
    elif units == 'real': # In kcal/mol
        energy = uc.convert(energy, 'kcal/mol', 'eV/atom', 'energy')
    return energy 

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
                            log_index = 0,  
                            energy_label='PotEng', 
                            units='metal'):
    data = {}
    for root, _, _ in os.walk(path_to_images):
        log_paths = glob.glob(os.path.join(root, '*.lammps'))
        for log_path in log_paths:
            try:
                energy = get_energy(log_path, log_index, energy_label, units=units)
            except:
                energy = np.nan
            p = Path(log_path)
            image = int(re.findall(r'\d+', p.name)[0])
            try:
                dump_path = glob.glob(os.path.join(root, f"*{image}*.dump"))[0]
                fxs, fys, fzs = get_forces(dump_path, units=units)
            except:
                fxs, fys, fzs = [], [], []

            md = p.parent.name
            ffield = p.parent.parent.name
            nested_set(data, [ffield, md, image], {'energy': energy, 
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
            nested_set(data, [ffield, md, image], {'energy': energy,
                                                   'fx': fxs, 'fy': fys, 'fz': fzs})
    return data

def comparison_dictionary(data_dictionary, ref_key='DFT', rel_image=None):
    dev_data = {}
    for ffield in list(data_dictionary.keys()):
        for md in list(data_dictionary[ffield].keys()):
            for image in list(data_dictionary[ffield][md].keys()):
                
                # Get energy difference first
                energy = data_dictionary[ffield][md][image]['energy']
                ref_energy = data_dictionary[ref_key][md][image]['energy']
                if rel_image is not None:
                    use_energy = np.subtract(energy, data_dictionary[ffield][md][rel_image]['energy'])
                    use_ref = np.subtract(ref_energy, data_dictionary[ref_key][md][rel_image]['energy']) 
                else:
                    use_energy = energy
                    use_ref = ref_energy
                energy_difference = np.abs(np.subtract(use_ref, use_energy))

                # Get force differences next
                ref_fx = data_dictionary[ref_key][md][image]['fx']
                fx_difference = np.subtract(ref_fx, data_dictionary[ffield][md][image]['fx'])
                fx_rmsd = np.sqrt(np.mean(fx_difference ** 2))
                ref_fy = data_dictionary[ref_key][md][image]['fy']
                fy_difference = np.subtract(ref_fy, data_dictionary[ffield][md][image]['fy'])
                fy_rmsd = np.sqrt(np.mean(fy_difference ** 2))
                ref_fz = data_dictionary[ref_key][md][image]['fz']
                fz_difference = np.subtract(ref_fz, data_dictionary[ffield][md][image]['fz'])
                fz_rmsd = np.sqrt(np.mean(fz_difference ** 2))

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

