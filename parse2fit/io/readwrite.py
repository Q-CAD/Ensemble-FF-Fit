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
        # Reax-specific configuration
        
        self.write_options = ['energy', 'charges', 'forces', 'distances', 
                              'angles', 'dihedrals', 'lattice_vectors']
        self.default_weights = self._set_default_weights(self.config_dct.get('default_weights'))
        self.generation_parameters = self.config_dct.get('generation_parameters')
        self.geo, self.trainsetin = self._load_input_files()
        self.input_paths = self.config_dct.get('input_paths')
    
    def _load_input_files(self):
        if 'input_directory' in self.generation_parameters:
            geo_path = os.path.join(self.generation_parameters['input_directory'], 'geo')
            trainsetin_path = os.path.join(self.generation_parameters['input_directory'], 'trainset.in')
            if os.path.exists(geo_path):
                geo = self.file_to_string(geo_path)
            else:
                print(f'input_directory tag supplied but {geo_path} does not exist! Using empty default')
                geo = ''
            if os.path.exists(trainsetin_path):
                trainsetin = self.file_to_string(trainsetin_path)
            else:
                print(f'input_directory tag supplied but {trainsetin_path} does not exist! Using empty default')
                trainsetin = ''
        else:
            geo, trainsetin = '', ''
        return geo, trainsetin

    def _set_default_weights(self, weight_dct):
        dct = {}
        default = {'split': 0.5, 'min': 0, 'max': 1, 'type': 'binary'}

        for prop in self.write_options:
            if prop in weight_dct:
                if isinstance(weight_dct[prop], dict):
                    dct[prop] = weight_dct[prop]
                else:
                    dct[prop] = default
            else:
                dct[prop] = default
        return dct 
    
    def _default_labeling(self, path, label, num=3):
        if num > 0:
            directory_split = os.path.split(path)
            new_directory = directory_split[0]
            if label == '':
                new_label = directory_split[1]
            else:
               new_label = directory_split[1] + '_' + label
            return self._default_labeling(new_directory, new_label, num-1)
        else:
            return label

    def _get_energy_paths(self):
        relative_energy_paths = []
        for path in self.input_paths:
            if 'energy' in self.input_paths[path]:
                if isinstance(self.input_paths[path]['energy'], dict):
                    if self.input_paths[path]['energy'].get('add') is not None:
                        relative_energy_paths += self.input_paths[path]['energy']['add']
                    if self.input_paths[path]['energy'].get('subtract') is not None:
                        relative_energy_paths += self.input_paths[path]['energy']['subtract']
        return relative_energy_paths

    def _get_path_values(self, top_path, root_path):
        # Based on the root path
        path_dct = {}
        supported_properties = ['label', 'structure'] + self.write_options 
        for prop in supported_properties:
            try:
                value = self.input_paths[top_path].get(prop)
            except KeyError: # Can arise for the energy_paths specified, or if value not specified
                value = None
            if prop == 'label':
                if value is None:
                    value = self._default_labeling(root_path, '') # Default 
            elif prop == 'structure': 
                if value is None:
                    value = {'pymatgen_structure': True} # Default
                elif isinstance(value, dict):
                    if value.get('pymatgen_structure') is None and value.get('ase_structure') is None:
                        value['pymatgen_structure'] = True # Default
                else:
                    print(f'{value} not recognized for {prop} in .yml; defaulting to pymatgen_structure=True')
                    value = {'pymatgen_structure': True}
            elif prop == 'energy': 
                if value is None or value is True:
                    value = {'weights': self.default_weights['energy']} # Default
                elif isinstance(value, dict): # Dictionary passed
                    if value.get('weights') is None: # No weights specified
                        value['weights'] = self.default_weights['energy']
                    elif isinstance(value.get('weights'), int):
                        value['weights'] = float(value['weights']) # Set fixed energy weighting here
                else:
                    print(f'{value} not recognized for {prop} in .yml; defaulting to True')
                    value = {'weights': self.default_weights['energy']} # Default
            else:
                if value is None or value is False:
                    pass
                elif value is True:
                    value = {'weights': self.default_weights[prop]}
                elif isinstance(value, dict):
                     if value.get('weights') is None: # No weights specified
                        value['weights'] = self.default_weights[prop]
                     elif isinstance(value.get('weights'), list):
                        value['weights'] = [float(val) for val in value['weights']]
                else:
                    print(f'{value} not recognized for {prop} in .yml; defaulting to False')
            path_dct[prop] = value
        path_dct['top_path'] = top_path
        return path_dct

    def _none_false(self, val):
        if val is None or val is False:
            return False
        else:
            return True

    # Multiprocessing parsing
    '''
    def _dct_parser(self, path_dct, path):
        # Simplified version of your existing _dct_parser function
        try:
            parsed_dct = ParserFactory.create_parser(path,
                pymatgen_structure=self._none_false(path_dct['structure'].get('pymatgen_structure')),
                ase_atoms=self._none_false(path_dct['structure'].get('ase_atoms')),
                periodic_geometries=self._none_false(self.generation_parameters.get('periodic_geometries')), 
                energy=self._none_false(path_dct.get('energy')),
                charges=self._none_false(path_dct.get('charges')),
                forces=self._none_false(path_dct.get('forces')),
                distances=self._none_false(path_dct.get('distances')),
                angles=self._none_false(path_dct.get('angles')),
                dihedrals=self._none_false(path_dct.get('dihedrals')),
                lattice_vectors=self._none_false(path_dct.get('lattice_vectors'))
                ).parse_all()
            reax_entry = ReaxEntry(label=path_dct['label'])
            reax_entry.from_dict(parsed_dct)
            #print(f'ReaxFF entry is: {reax_entry}')
            path_dct['reax_entry'] = reax_entry
            return path_dct
        except AttributeError:
            return None

    def _get_all_paths(self, paths, walk_roots):
        all_paths = [] # top path, root path or top path, top path
        for path in paths:
            if walk_roots:
                for root, _, _ in os.walk(os.path.abspath(path), topdown=True):
                    all_paths.append((path, root))
            else:
                all_paths.append((path, path))
        #print(all_paths)
        return all_paths

    def _worker_task(self, path):
        try:
            path_dct = self._get_path_values(top_path=path[0], root_path=path[1])  # Dictionary associated with each path
            parsed_dct = self._dct_parser(path_dct, path[1])
            return (path[1], parsed_dct) # Returns the parsed dictionary
        except (ValueError, AttributeError):  # Handle errors
            return None

    def _build_objects_dct(self, objects_dct, paths, walk_roots=True):
        # Step 1: Gather all paths
        all_paths = self._get_all_paths(paths, walk_roots)

        # Step 2: Parallelize parsing using multiprocessing
        with Pool() as pool:
            results = pool.map(self._worker_task, all_paths)
        
        # Step 3: Aggregate results
        for i, result in enumerate(results):
            if result[1]:  # Ignore null values for object construction
                if all_paths[i][1] not in list(objects_dct.keys()):
                    objects_dct[result[0]] = result[1]

        return objects_dct
    '''
    # Serialized parsing
    #'''
    def _dct_parser(self, objects_dct, path_dct, path):
        parsed_dct = ParserFactory.create_parser(path,
                pymatgen_structure=self._none_false(path_dct['structure'].get('pymatgen_structure')),
                ase_atoms=self._none_false(path_dct['structure'].get('ase_atoms')),
                periodic_geometries=self._none_false(self.generation_parameters.get('periodic_geometries')), 
                energy=self._none_false(path_dct.get('energy')),
                charges=self._none_false(path_dct.get('charges')),
                forces=self._none_false(path_dct.get('forces')),
                distances=self._none_false(path_dct.get('distances')),
                angles=self._none_false(path_dct.get('angles')),
                dihedrals=self._none_false(path_dct.get('dihedrals')),
                lattice_vectors=self._none_false(path_dct.get('lattice_vectors'))
                ).parse_all()
        reax_entry = ReaxEntry(label=path_dct['label'])
        reax_entry.from_dict(parsed_dct)
        path_dct['reax_entry'] = reax_entry
        objects_dct[path] = path_dct
        return objects_dct

    def _build_objects_dct(self, objects_dct, paths, walk_roots=True):
        for path in paths:
            if walk_roots == True:
                for root, dirs, files in os.walk(os.path.abspath(path), topdown=True):
                    if root not in list(objects_dct.keys()):
                        path_dct = self._get_path_values(top_path=path, root_path=root) # Dictionary associated with each path
                        try:
                            objects_dct = self._dct_parser(objects_dct, path_dct, root)
                        except (ValueError, AttributeError):
                            continue
            else:
                if path not in list(objects_dct.keys()):
                    path_dct = self._get_path_values(top_path=path, root_path=path) # Dictionary associated with each path
                    try:
                        objects_dct = self._dct_parser(objects_dct, path_dct, path)
                    except (ValueError, AttributeError): # No DFT code in the root directory being searched
                        continue
        return objects_dct
    #'''

    def _check_paths(self, paths_list):
        for path in paths_list:
            if os.path.exists(path) is False:
                print(f'{path} does not exist; check input .yml!')
                sys.exit(1)
        return 

    def _construct_objects_dct(self):
        sorted_input_paths = sorted(list(self.input_paths.keys()), key=len, reverse=True)
        sorted_energy_paths = sorted(list(self._get_energy_paths()), key=len, reverse=True)
        self._check_paths(sorted_input_paths)
        self._check_paths(sorted_energy_paths)
        objects_dct = self._build_objects_dct({}, sorted_input_paths) # Construct top level paths first
        objects_dct = self._build_objects_dct(objects_dct, sorted_energy_paths, walk_roots=False) # Then attempt to construct energy paths
        return objects_dct

    def _get_relative_energies(self, objects_dictionary):
        ''' Continue editing this ''' 
        # Compute weights across all relative energies
        energy_dct = {} # ReaxObj: {relative_energy: float, add: [], subtract: [], divisors: , weight, sig_figs}
        for object_path in objects_dictionary:
            reax_obj = objects_dictionary[object_path]['reax_entry']
            add_reax_objs, subtract_reax_objs = [], []
            if 'add' in objects_dictionary[object_path]['energy']:
                for add_path in objects_dictionary[object_path]['energy']['add']:
                    add_reax_objs.append(objects_dictionary[add_path]['reax_entry'])
            if 'subtract' in objects_dictionary[object_path]['energy']:
                for subtract_path in objects_dictionary[object_path]['energy']['subtract']:
                    subtract_reax_objs.append(objects_dictionary[subtract_path]['reax_entry'])
            
            if add_reax_objs == [] and subtract_reax_objs == []:
                pass
            else:
                energy_dct[object_path] = {}
                if 'get_divisors' in objects_dictionary[object_path]['energy']:
                    get_divisors = objects_dictionary[object_path]['energy']['get_divisors']
                else:
                    get_divisors = False # Default is to just use 1 for everything, default within the ReaxEntry
                reax_objs, signs, divisors, relative_energy = reax_obj.get_relative_energy(add_reax_objs, 
                                                                                               subtract_reax_objs, 
                                                                                               get_divisors)
                energy_dct[object_path] = {'reax_entry': reax_obj,
                                           'relative_energy': relative_energy.value,
                                           'add': add_reax_objs, 
                                           'subtract': subtract_reax_objs, 
                                           'get_divisors': get_divisors}
        
        ### Get the weights to use from the relative energy values
        super_paths, super_values = [], []
        for energy_path in list(energy_dct.keys()):
            relative_energy = energy_dct[energy_path]['relative_energy']
            weights = objects_dictionary[energy_path]['energy']['weights']
            if weights == self.default_weights['energy']: # Top level relative energy weights
                super_paths.append(energy_path)
                super_values.append(energy_dct[energy_path]['relative_energy'])
            elif isinstance(weights, float):
                energy_dct[energy_path]['weight'] = weights
            elif isinstance(weights, dict): # Different dictionary
                # WARNING: Doesn't work for magnitude weighting within the same path !!!
                energy_dct[energy_path]['weight'] = WeightedSampler([relative_energy], weights).sample()[0]
        
        super_weights = WeightedSampler(super_values, self.default_weights['energy']).sample()
        for i, super_path in enumerate(super_paths):
            if isinstance(super_weights, float): # Only float returned
                energy_dct[super_path]['weight'] = super_weights
            else:
                energy_dct[super_path]['weight'] = super_weights[i]
        return energy_dct
    
    def get_geo_string(self, objects_dictionary):
        write_string = deepcopy(self.geo) # Copy the geo file string
        for path_i, object_path in enumerate(list(objects_dictionary.keys())):
            reax_obj = objects_dictionary[object_path]['reax_entry']
            structure_args = objects_dictionary[object_path]['structure']
            reax_obj_str = reax_obj.structure_to_string(**structure_args)
            if reax_obj_str not in write_string: # Check to see if full string is present
                write_string += reax_obj_str
        return write_string

    def _modify_string(self, main_string, check_string, substitute_string, insert_point_string=None, placeholder='weight', pattern=r"(\d+\.\d*)"):
        ''' Check for regex pattern, and substitute string into existing string '''
        escaped_check_string = re.escape(check_string).replace(placeholder, pattern)
        escaped_pattern = re.compile(escaped_check_string)
        match = escaped_pattern.search(main_string)
        if match:
            if 'fix_input_weights' in self.generation_parameters and self.generation_parameters['fix_input_weights'] is True:
                #print(f'Input weights are fixed; no modifications to be made to {match.group()}')
                return main_string # Keep everything fixed, do not change
            else:
                if match.group() != substitute_string: # If True, not a header or footer
                    weight_match = escaped_pattern.search(substitute_string)
                    weight = float(weight_match.group(1)) # Get the weight from the match
                    if weight == 0: # Remove zero-valued weight strings from main_string
                        return main_string.replace(match.group(), '')
                    else:
                        return main_string.replace(match.group(), substitute_string) # Replace old string with new string
                else:
                    return main_string # Same string is matched, so no changes necessary
        else:
            if check_string == substitute_string: # Catches headers, footers
                return main_string + substitute_string # Add to end
            else:
                weight_match = escaped_pattern.search(substitute_string)
                weight = float(weight_match.group(1)) # Get the weight from the match
                if weight == 0: # Remove zero-valued weight strings from main_string
                    return main_string
                else:
                    if insert_point_string: # Insert after the insert_point_string variable
                        position = main_string.find(insert_point_string)
                        if position != -1: # Finds insert_point_string
                            insertion_point = position + len(insert_point_string)
                            return main_string[:insertion_point] + substitute_string + main_string[insertion_point:]
                        else:
                            return main_string + substitute_string
                    else: # No insert_point_string
                        return main_string + substitute_string

    def _split_lines_and_modify(self, main_string, regex_string, weights_string, insert_point_string=None, placeholder='weight', pattern=r"(\d+\.\d*)"):
        split_regex_strings = re.split(r'(\n)', regex_string)
        regex_list = [split_regex_strings[i] + split_regex_strings[i + 1] for i in range(0, len(split_regex_strings) - 1, 2)]
        split_weights_strings = re.split(r'(\n)', weights_string)
        weights_list = [split_weights_strings[i] + split_weights_strings[i + 1] for i in range(0, len(split_weights_strings) - 1, 2)]
        #print(regex_list, weights_list)
        for i, regex in enumerate(regex_list):
            main_string = self._modify_string(main_string, regex, weights_list[i], insert_point_string, placeholder, pattern)
        return main_string

    def get_trainsetin_string(self, objects_dictionary, self_energy=False):
        write_string = deepcopy(self.trainsetin)
        ere = ReaxEntry(label='empty') # For writing headers and footers
        
        # Handling non-Geometry properties
        for write_option in ['charges', 'forces', 'lattice_vectors']:
            property_start = ere.trainsetin_section_header(write_option)
            write_string = self._modify_string(write_string, property_start, property_start) # Add start if it doesn't exist

            for path_i, object_path in enumerate(list(objects_dictionary.keys())):
                option_val = objects_dictionary[object_path][write_option]
                if isinstance(option_val, dict):
                    weights = option_val['weights']
                    reax_obj = objects_dictionary[object_path]['reax_entry']
                    re_string = reax_obj.get_property_string(write_option, weights='weight',
                                                                 default_weights=self.default_weights[write_option]) # For regex
                    weight_string = reax_obj.get_property_string(write_option, weights=weights,
                                                                 default_weights=self.default_weights[write_option])
                    write_string = self._split_lines_and_modify(write_string, re_string, weight_string, property_start)
            property_end = ere.trainsetin_section_footer(write_option)
            write_string = self._modify_string(write_string, property_end, property_end) 
        
        # Handling Geometry Properties
        geometry_start = ere.trainsetin_section_header('distances')
        write_string = self._modify_string(write_string, geometry_start, geometry_start) # Add start if it doesn't exist

        for write_option in ['distances', 'angles', 'dihedrals']:
            for path_i, object_path in enumerate(list(objects_dictionary.keys())):
                option_val = objects_dictionary[object_path][write_option]
                if isinstance(option_val, dict):
                    weights = objects_dictionary[object_path][write_option]['weights']
                    reax_obj = objects_dictionary[object_path]['reax_entry']
                    re_string = reax_obj.get_property_string(write_option, weights='weight',
                                                                 default_weights=self.default_weights[write_option]) # For regex
                    weight_string = reax_obj.get_property_string(write_option, weights=weights,
                                                                 default_weights=self.default_weights[write_option])
                    write_string = self._split_lines_and_modify(write_string, re_string, weight_string, geometry_start)
        geometry_end = ere.trainsetin_section_footer('distances')
        write_string = self._modify_string(write_string, geometry_end, geometry_end)

        # Handling energy
        re_dictionary = self._get_relative_energies(objects_dictionary)
        energy_start = ere.trainsetin_section_header('energy')
        write_string = self._modify_string(write_string, energy_start, energy_start) # Add start if it doesn't exist

        for path_j, object_path in enumerate(list(re_dictionary.keys())):
            reax_obj = re_dictionary[object_path]['reax_entry']
            weight = re_dictionary[object_path]['weight']
            add = re_dictionary[object_path]['add']
            subtract = re_dictionary[object_path]['subtract']
            if self_energy == False and (reax_obj in add or reax_obj in subtract):
                continue # Don't write self energy
            else:
                re_string = reax_obj.relative_energy_to_string(weight='weight', add=add, subtract=subtract,
                                                               get_divisors=re_dictionary[object_path]['get_divisors'])
                weight_string = reax_obj.relative_energy_to_string(weight=weight, add=add, subtract=subtract,
                                                               get_divisors=re_dictionary[object_path]['get_divisors'])
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
        #print(objects_dictionary)
        geo_string = self.get_geo_string(objects_dictionary)

        runs_to_generate = self.generation_parameters['runs_to_generate']
        output_directory = self.generation_parameters['output_directory']

        trainsetin_strings = []
        unique = 0
        attempts = 0
        
        print('Writing unique trainset.in files...')
        while unique < runs_to_generate: # Get unique trainset.in files
            if attempts > attempt_multiplier * runs_to_generate:
                print(f'Number of unique trainset.in file generation attempts exceeds {attempt_multiplier * runs_to_generate}; breaking loop...')
                break
            trainsetin_string = self.get_trainsetin_string(objects_dictionary, self_energy)
            if trainsetin_string not in trainsetin_strings:
                subdirectory = self.config_dct.get('output_format') + '_run_' + str(unique)
                write_path = os.path.join(output_directory, subdirectory)
                os.makedirs(write_path, exist_ok=True)
                self.string_to_file(os.path.join(write_path, 'geo'), geo_string)
                self.string_to_file(os.path.join(write_path, 'trainset.in'), trainsetin_string)
                trainsetin_strings.append(trainsetin_string)
                unique += 1
            attempts += 1
        print(f'Finished successfully, {unique} unique trainset.in files written to {output_directory}.')
        return
