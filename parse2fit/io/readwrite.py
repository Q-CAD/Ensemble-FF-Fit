from abc import ABC, abstractmethod
from parse2fit.core.entries import ReaxEntry
from parse2fit.io.parsers import ParserFactory
from parse2fit.tools.weights import WeightedSampler
import warnings
import yaml
from pathlib import Path
import os
import sys
from multiprocessing import Pool
from copy import deepcopy
import re
import logging

warnings.filterwarnings('ignore')

class YamlManager(ABC):
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)

class ReadWriteFactory(YamlManager):
    def __init__(self, config_path):
        super().__init__(config_path)

    def get_writer(self):
        if not self.config.get('output_format'):
            raise ValueError(f"No 'output_format' found in {self.config_path}")
        elif self.config.get('output_format') == 'jax-reaxff':
            return ReaxRW(self.config)

    def write_output(self, yaml_path):
        pass

class ReaxRW():
    def __init__(self, config_dct):
        self.config_dct = config_dct
        self.write_options = ['energy', 'charges', 'forces', 'distances', 
                              'angles', 'dihedrals', 'lattice_vectors']
        self.default_weights = self._set_default_weights(self.config_dct.get('default_weights', {}))
        self.generation_parameters = self.config_dct.get('generation_parameters', {})
        self.geo, self.trainsetin = self._load_input_files()
        self.input_paths = self.config_dct.get('input_paths', {})

    def _load_input_files(self):
        """Loads geo and trainset.in files if input_directory is specified."""
        input_dir = self.generation_parameters.get('input_directory', '')
        geo_path = os.path.join(input_dir, 'geo')
        trainsetin_path = os.path.join(input_dir, 'trainset.in')

        def read_file(path, default_value=''):
            if os.path.exists(path):
                return self.file_to_string(path)
            logging.warning(f"File {path} not found! Using empty default.")
            return default_value

        return read_file(geo_path), read_file(trainsetin_path)

    def _set_default_weights(self, weight_dct):
        """Sets default weights for various properties."""
        default = {'split': 0.5, 'min': 0, 'max': 1, 'type': 'binary'}
        return {prop: weight_dct.get(prop, default) if isinstance(weight_dct.get(prop), dict) else default
                for prop in self.write_options}

    def _default_labeling(self, path, label='', depth=3):
        """Generates default labels by iterating up the directory hierarchy."""
        if depth == 0:
            return label
        new_label = os.path.basename(path) if not label else f"{os.path.basename(path)}_{label}"
        return self._default_labeling(os.path.dirname(path), new_label, depth - 1)

    def _get_energy_paths(self):
        """Extracts all energy-related paths from the input_paths dictionary."""
        relative_energy_paths = []
        for path_config in self.input_paths.values():
            energy_config = path_config.get('energy', {})
            if isinstance(energy_config, dict):
                relative_energy_paths.extend(energy_config.get('add', []))
                relative_energy_paths.extend(energy_config.get('subtract', []))
        return relative_energy_paths

    def _get_path_values(self, top_path, root_path):
        """Extracts and processes path-related configuration settings."""
        path_config = self.input_paths.get(top_path, {})
        path_dct = {}
        supported_properties = ['label', 'structure'] + self.write_options

        for prop in supported_properties:
            value = path_config.get(prop)

            if prop == 'label':
                path_dct[prop] = value if value is not None else self._default_labeling(root_path)

            elif prop == 'structure':
                if value is None:
                    path_dct[prop] = {'pymatgen_structure': True}
                elif isinstance(value, dict):
                    if 'pymatgen_structure' not in value and 'ase_structure' not in value:
                        value['pymatgen_structure'] = True
                    path_dct[prop] = value
                else:
                    logging.warning(f"Invalid value {value} for {prop}; defaulting to pymatgen_structure=True")
                    path_dct[prop] = {'pymatgen_structure': True}

            elif prop == 'energy':
                if value is None or value is True:
                    path_dct[prop] = {'weights': self.default_weights['energy']}
                elif isinstance(value, dict):
                    value.setdefault('weights', self.default_weights['energy'])
                    if isinstance(value['weights'], int):
                        value['weights'] = float(value['weights'])
                    path_dct[prop] = value
                else:
                    logging.warning(f"Invalid value {value} for {prop}; defaulting to True")
                    path_dct[prop] = {'weights': self.default_weights['energy']}

            else:
                if value is None or value is False:
                    path_dct[prop] = False
                elif value is True:
                    path_dct[prop] = {'weights': self.default_weights[prop]}
                elif isinstance(value, dict):
                    value.setdefault('weights', self.default_weights[prop])
                    if isinstance(value['weights'], list):
                        value['weights'] = [float(val) for val in value['weights']]
                    path_dct[prop] = value
                else:
                    logging.warning(f"Invalid value {value} for {prop}; defaulting to False")
                    path_dct[prop] = False
        path_dct['top_path'] = top_path
        return path_dct
    
    def _dct_parser(self, path_dct, path):
        
        def true_if(val):
            if val is None or val is False:
                return False
            else:
                return True

        kwargs_dct = {'pymatgen_structure': true_if(path_dct['structure'].get('pymatgen_structure')), 
                      'ase_atoms': true_if(path_dct['structure'].get('ase_atoms')), 
                      'periodic_geometries': true_if(self.generation_parameters.get('periodic_geometries')), 
                      'energy': true_if(path_dct.get('energy')),
                      'charges': true_if(path_dct.get('charges')),
                      'forces': true_if(path_dct.get('forces')),
                      'distances': true_if(path_dct.get('distances')),
                      'angles': true_if(path_dct.get('angles')),
                      'dihedrals': true_if(path_dct.get('dihedrals')),
                      'lattice_vectors': true_if(path_dct.get('lattice_vectors'))
                    }
        parsed_dct = ParserFactory.create_parser(path, **kwargs_dct)
        
        if parsed_dct:
            reax_entry = ReaxEntry(label=path_dct['label'])
            reax_entry.from_dict(parsed_dct.parse_all())
        else:
            reax_entry = None
        path_dct['reax_entry'] = reax_entry
            
        return path_dct
        
    def _get_all_paths(self, paths, walk_roots):
        all_paths = [] # top path, root path or top path, top path
        for path in paths:
            if walk_roots:
                for root, _, _ in os.walk(os.path.abspath(path), topdown=True):
                    if root not in [pair[1] for pair in all_paths]:
                        all_paths.append((path, root))
            else:
                if path not in [pair[1] for pair in all_paths]:
                    all_paths.append((path, path))
        
        return all_paths

    def _worker_task(self, path):
        path_dct = self._get_path_values(top_path=path[0], root_path=path[1])  # Dictionary associated with each path
        reax_entry = self._dct_parser(path_dct, path[1])
        
        return (path[1], reax_entry) # Returns the parsed ReaxEntry or None

    def _build_objects_dct(self, objects_dct, paths, walk_roots=True):
        # Step 1: Gather all paths
        all_paths = self._get_all_paths(paths, walk_roots)
        
        # Step 2: Parallelize parsing using multiprocessing
        with Pool() as pool:
            results = pool.map(self._worker_task, all_paths)
        
        # Step 3: Aggregate results
        for i, result in enumerate(results):
            if result[1]['reax_entry']:  # Ignore Nonetype Reax objects
                if all_paths[i][1] not in list(objects_dct.keys()):
                    objects_dct[result[0]] = result[1]

        return objects_dct

    def _check_paths(self, paths_list):
        for path in paths_list:
            if os.path.exists(path) is False:
                print(f'{path} does not exist; check input .yml!')
                sys.exit(1)
        return 

    def _construct_objects_dct(self):
        sorted_input_paths = sorted(list(set(self.input_paths.keys())), key=len, reverse=True)
        sorted_energy_paths = sorted(list(set(self._get_energy_paths())), key=len, reverse=True)

        self._check_paths(sorted_input_paths)
        self._check_paths(sorted_energy_paths)
        
        objects_dct = self._build_objects_dct({}, sorted_input_paths) # Construct top level paths first
        all_sorted_input_paths = sorted(list(objects_dct.keys()))
        
        parse_sorted_energy_paths = [path for path in sorted_energy_paths if path not in all_sorted_input_paths]
        objects_dct = self._build_objects_dct(objects_dct, parse_sorted_energy_paths, walk_roots=False) # Then attempt to construct energy paths
         
        return objects_dct

    def _extract_relative_energy_components(self, obj_dict, obj_path):
        """ Extracts add, subtract, and divisor entries for a given object. """
        entry = obj_dict[obj_path]['reax_entry']
        add_entries = [obj_dict[path]['reax_entry'] for path in obj_dict[obj_path]['energy'].get('add', [])]
        subtract_entries = [obj_dict[path]['reax_entry'] for path in obj_dict[obj_path]['energy'].get('subtract', [])]
        divisors = obj_dict[obj_path]['energy'].get('get_divisors', False)

        return entry, add_entries, subtract_entries, divisors

    def _compute_relative_energy(self, obj_dict):
        """ Computes relative energies for each object and stores in a dictionary. """
        energy_dict = {}

        for obj_path in obj_dict:
            entry, add_entries, subtract_entries, divisors = self._extract_relative_energy_components(obj_dict, obj_path)

            if not add_entries and not subtract_entries:
                continue  # No relative energy calculation needed

            reax_objs, signs, divs, rel_energy = entry.get_relative_energy(add_entries, subtract_entries, divisors)

            energy_dict[obj_path] = {
                'reax_entry': entry,
                'relative_energy': rel_energy.value,
                'add': add_entries,
                'subtract': subtract_entries,
                'get_divisors': divisors
            }

        return energy_dict

    def _assign_weights(self, energy_dict, obj_dict):
        """ Assigns weights to relative energy values. """
        super_paths, super_values = [], []

        for energy_path, data in energy_dict.items():
            rel_energy = data['relative_energy']
            weights = obj_dict[energy_path]['energy']['weights']

            if weights == self.default_weights['energy']:
                super_paths.append(energy_path)
                super_values.append(rel_energy)
            elif isinstance(weights, float):
                data['weight'] = weights
            elif isinstance(weights, dict):
                data['weight'] = WeightedSampler([rel_energy], weights).sample()[0]

        super_weights = WeightedSampler(super_values, self.default_weights['energy']).sample()

        for i, super_path in enumerate(super_paths):
            energy_dict[super_path]['weight'] = super_weights if isinstance(super_weights, float) else super_weights[i]

        return energy_dict

    def _get_relative_energies(self, objects_dictionary):
        """ Wrapper function to compute relative energies and assign weights. """
        energy_dict = self._compute_relative_energy(objects_dictionary)
        return self._assign_weights(energy_dict, objects_dictionary)
    
    def get_geo_string(self, objects_dictionary):
        write_string = deepcopy(self.geo) # Copy the geo file string
        for path_i, object_path in enumerate(list(objects_dictionary.keys())):
            reax_obj = objects_dictionary[object_path]['reax_entry']
            structure_args = objects_dictionary[object_path]['structure']
            reax_obj_str = reax_obj.structure_to_string(**structure_args)
            if reax_obj_str not in write_string: # Check to see if full string is present
                write_string += reax_obj_str
        return write_string

    def _modify_string(self, main_str, check_str, sub_str, insert_point_str=None, placeholder='weight', pattern=r"(\d+\.\d*)"):
        """ Modifies a string by replacing or inserting values based on regex matching. """

        # Escape special regex characters and replace placeholder
        regex_pattern = re.escape(check_str).replace(placeholder, pattern)
        compiled_pattern = re.compile(regex_pattern)

        match = compiled_pattern.search(main_str)

        if match:
            if self.generation_parameters.get('fix_input_weights', True):
                return main_str  # No modifications if weights are fixed

            if match.group() != sub_str:  # If actual replacement is needed
                weight_match = compiled_pattern.search(sub_str)
                weight = float(weight_match.group(1)) if weight_match else None

                if weight == 0:
                    return main_str.replace(match.group(), '')  # Remove zero-weight entries
                else:
                    return main_str.replace(match.group(), sub_str)  # Replace with new string
            return main_str  # No change needed
        else:
            # Handle case where check_str isn't found in main_str
            weight_match = compiled_pattern.search(sub_str)
            try:
                weight = float(weight_match.group(1))
            except (AttributeError, IndexError):
                weight = None

            if weight == 0:
                return main_str  # Don't insert zero-weight strings

            if insert_point_str:
                pos = main_str.find(insert_point_str)
                if pos != -1:
                    return main_str[:pos + len(insert_point_str)] + sub_str + main_str[pos + len(insert_point_str):]
            return main_str + sub_str  # Default append if no insert point

    def _split_lines_and_modify(self, main_str, regex_str, weights_str, insert_point_str=None, placeholder='weight', pattern=r"(\d+\.\d*)"):
        """ Splits strings line by line and modifies matching sections. """

        regex_list = [r + n for r, n in zip(re.split(r'(\n)', regex_str)[::2], re.split(r'(\n)', regex_str)[1::2])]
        weights_list = [w + n for w, n in zip(re.split(r'(\n)', weights_str)[::2], re.split(r'(\n)', weights_str)[1::2])]

        for regex, weight in zip(regex_list, weights_list):
            main_str = self._modify_string(main_str, regex, weight, insert_point_str, placeholder, pattern)

        return main_str

    def get_trainsetin_string(self, objects_dictionary, self_energy=False):
        write_string = deepcopy(self.trainsetin)
        ere = ReaxEntry(label='empty')

        def update_write_string(write_string, ere, section, obj_dict, header=True, footer=True):
            start_marker = ere.trainsetin_section_header(section)
            end_marker = ere.trainsetin_section_footer(section)

            if header:
                write_string = self._modify_string(write_string, start_marker, start_marker)

            for object_path, obj_data in obj_dict.items():
                option_val = obj_data.get(section)
                if isinstance(option_val, dict):
                    weights = option_val['weights']
                    reax_obj = obj_data['reax_entry']

                    re_string = reax_obj.get_property_string(section, weights='weight',
                                                         default_weights=self.default_weights[section])
                    weight_string = reax_obj.get_property_string(section, weights=weights,
                                                             default_weights=self.default_weights[section])
                    write_string = self._split_lines_and_modify(write_string, re_string, weight_string, start_marker)

            if footer:
                write_string = self._modify_string(write_string, end_marker, end_marker)
            return write_string

        # Process non-Geometry properties
        for prop in ['charges', 'forces', 'lattice_vectors']:
            write_string = update_write_string(write_string, ere, prop, objects_dictionary, header=True, footer=True)

        # Process Geometry properties
        geometry_start, geometry_end = ere.trainsetin_section_header('distances'), ere.trainsetin_section_footer('distances')
        write_string = self._modify_string(write_string, geometry_start, geometry_start)
        geometry_properties = ['distances', 'angles', 'dihedrals']
        for prop in geometry_properties:
            write_string = update_write_string(write_string, ere, prop, objects_dictionary, header=False, footer=False)
        write_string = self._modify_string(write_string, geometry_end, geometry_end)

        # Process Energy
        re_dictionary = self._get_relative_energies(objects_dictionary)
        energy_start = ere.trainsetin_section_header('energy')
        write_string = self._modify_string(write_string, energy_start, energy_start)

        for object_path, energy_data in re_dictionary.items():
            reax_obj = energy_data['reax_entry']
            weight = energy_data['weight']
            add = energy_data['add']
            subtract = energy_data['subtract']

            if not self_energy and (reax_obj in add or reax_obj in subtract):
                continue  # Skip self-energy entries

            re_string = reax_obj.relative_energy_to_string(weight='weight', add=add, subtract=subtract,
                                                        get_divisors=energy_data['get_divisors'])
            weight_string = reax_obj.relative_energy_to_string(weight=weight, add=add, subtract=subtract,
                                                            get_divisors=energy_data['get_divisors'])
            write_string = self._modify_string(write_string, re_string, weight_string, energy_start)

        energy_end = ere.trainsetin_section_footer('energy')
        write_string = self._modify_string(write_string, energy_end, energy_end)

        return write_string

    def file_to_string(self, filepath):
        with open(filepath, "r") as text_file:
            data = text_file.read()
        return data

    def string_to_file(self, filepath, write_string):
        with open(filepath, "w") as text_file:
            text_file.write(write_string)
        return 
    
    def write_trainsetins(self, attempt_multiplier=5, self_energy=False):
        print('Constructing ReaxFF objects dictionary...')
        objects_dictionary = self._construct_objects_dct()
        geo_string = self.get_geo_string(objects_dictionary)

        runs_to_generate = self.generation_parameters['runs_to_generate']
        output_directory = self.generation_parameters['output_directory']

        unique_trainset_strings = set()  # Use a set instead of a list for fast lookups
        attempts = 0
        max_attempts = attempt_multiplier * runs_to_generate

        print('Writing unique trainset.in files...')

        while len(unique_trainset_strings) < runs_to_generate:
            if attempts > max_attempts:
                print(f'Number of unique trainset.in file generation attempts exceeds {max_attempts}; breaking loop...')
                break

            trainsetin_string = self.get_trainsetin_string(objects_dictionary, self_energy)

            if trainsetin_string not in unique_trainset_strings:
                unique_trainset_strings.add(trainsetin_string)  # Add to the set

                unique_index = len(unique_trainset_strings) - 1
                subdirectory = f"{self.config_dct.get('output_format')}_run_{unique_index}"
                write_path = os.path.join(output_directory, subdirectory)
                os.makedirs(write_path, exist_ok=True)

                self.string_to_file(os.path.join(write_path, 'geo'), geo_string)
                self.string_to_file(os.path.join(write_path, 'trainset.in'), trainsetin_string)

            attempts += 1

        print(f'Finished successfully, {len(unique_trainset_strings)} unique trainset.in files written to {output_directory}.')
