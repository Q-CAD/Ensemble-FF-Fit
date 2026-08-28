import yaml
import re
import json
import os
import math
from pymatgen.core import Structure
import numpy as np
from EnsembleFFFit.density_functional_theory.rmg.valence import ONCVValences, GeneralValences
from EnsembleFFFit.density_functional_theory.rmg.processor_grid import get_processor_grid

# Conversion factor from Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.529177


def _round_sig(number, sig=3):
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - sig))


def _parse_map(text: str) -> dict[str, str]:
    tokens = text.split()
    if len(tokens) % 2 != 0:
        raise ValueError("Expected an even number of tokens (key/value pairs) from 'pseudopotential' input parameter")
    # take every even token as a key, the following one as its value
    return dict(zip(tokens[0::2], tokens[1::2]))


def _sum_electrons(structure, pseudopotentials_directory, pseudo_dct):
    """
    Unlike the pyRMG original, this raises immediately (rather than printing
    and sys.exit(1)-ing) if any element's valence electron count can't be
    resolved -- a bare sys.exit() inside code that now runs as a MatEnsemble
    chore would kill the whole worker process rather than being reported as
    a clean, recorded chore failure.
    """
    if pseudopotentials_directory == '':
        valence = ONCVValences()
    else:
        valence = GeneralValences(pseudopotentials_directory, pseudo_dct)

    valences = [valence.get_valence(str(site.specie)) for site in structure]
    missing = sorted({str(site.specie) for site, v in zip(structure, valences) if v is None})
    if missing:
        raise ValueError(
            f"No valence electron count available for element(s) {missing} "
            f"(pseudopotentials_directory={pseudopotentials_directory!r})."
        )
    return np.sum(valences)


def _generate_wavefunction_grid(structure, cutoff, grid_divisibility_exponent):
    rca = np.pi / np.sqrt(cutoff) * BOHR_TO_ANGSTROM
    nx, ny, nz = np.rint(structure.lattice.abc / rca).astype(int)

    def grid_spacing_factors(nx, ny, nz, factor):
        return [(dim + factor - 1) // factor * factor for dim in [nx, ny, nz]]

    def anisotropy_check(structure, nxg, nyg, nzg):
        h_max = np.max(np.divide(structure.lattice.abc, [nxg, nyg, nzg]))
        h_min = np.min(np.divide(structure.lattice.abc, [nxg, nyg, nzg]))
        return h_max / h_min <= 1.1

    use_i = 1
    last_grid = (nx, ny, nz)
    for _ in range(grid_divisibility_exponent):
        use_i *= 2
        nxg, nyg, nzg = grid_spacing_factors(nx, ny, nz, use_i)
        if anisotropy_check(structure, nxg, nyg, nzg):
            last_grid = (nxg, nyg, nzg)
    return " ".join(str(n) for n in last_grid)


def _generate_kpoint_mesh(structure, kdelt):
    kpoints = [int(max(1, np.rint(np.divide(mag, kdelt)))) for mag in np.multiply(structure.lattice.reciprocal_lattice.abc, BOHR_TO_ANGSTROM)]
    return " ".join([str(k) for k in kpoints])


def compute_grid_and_resources(structure, input_args, target_nodes=0, gpus_per_node=8,
                                electrons_per_gpu=10, grid_divisibility_exponent=3,
                                pseudopotentials_directory='', pseudo_dct=None,
                                kpoint_multiplier=1):
    """
    The composition/lattice-dependent (never atomic-position-dependent) core
    of what RMGInput.from_yaml used to do in one shot: wavefunction grid,
    kpoint mesh/distribution, convergence-criterion scaling, electron
    counting, and processor grid + target_nodes sizing. Callable standalone
    so a caller can get a resource estimate (e.g. DFTMatEnsemble.build_dft_dcts,
    at task-construction time, before any chore is submitted) using whatever
    structure is on hand -- and the same function gets called again later,
    against the possibly-updated structure right before a run actually
    happens, to get the authoritative value. The two are only guaranteed to
    agree if the lattice itself hasn't changed between calls (e.g. under
    cell_relax); see RMG.write_input's consistency check for what happens
    when they don't.

    Returns (updated_input_args, target_nodes). Does not mutate the input_args
    dict passed in.
    """
    input_args = dict(input_args)
    pseudo_dct = dict(pseudo_dct) if pseudo_dct else {}

    # Unlike 'pseudo_dir'/'pseudopotential' (genuine RMG input keywords, kept
    # in input_args so they end up in the generated rmg_input file), none of
    # these are RMG keywords -- they're ensemble-orchestration knobs this
    # codebase invented, which test/RMG_testing/rmg_dft.py reads out of the
    # same yaml file. Popping them here (rather than leaving that to each
    # caller) means a value embedded directly in the yaml always overrides
    # whatever default the caller passed in (for the three this function
    # itself uses), and none of the five ever leaks into the generated
    # rmg_input file as a bogus keyword.
    gpus_per_node = input_args.pop('gpus_per_node', gpus_per_node)
    electrons_per_gpu = input_args.pop('electrons_per_gpu', electrons_per_gpu)
    grid_divisibility_exponent = input_args.pop('grid_divisibility_exponent', grid_divisibility_exponent)
    for orchestration_only_key in ('rmg_name', 'rmg_executable', 'command', 'structure_filename', 'allocated_nodes'):
        input_args.pop(orchestration_only_key, None)

    if 'cutoff' in input_args:
        wavefunction_grid = _generate_wavefunction_grid(structure, input_args['cutoff'], grid_divisibility_exponent)
        input_args['wavefunction_grid'] = wavefunction_grid
        input_args.pop('cutoff', 0)
    elif 'wavefunction_grid' in input_args:
        wavefunction_grid = input_args['wavefunction_grid']
    else:
        raise KeyError('Input .yml must contain "cutoff" or "wavefunction_grid"')

    if 'kdelt' in input_args:
        kpoint_mesh = _generate_kpoint_mesh(structure, input_args['kdelt'])
        input_args['kpoint_mesh'] = kpoint_mesh
        input_args.pop('kdelt', 0)
    elif 'kpoint_mesh' in input_args:
        pass
    else:
        raise KeyError('Input .yml must contain "kdelt" or "kpoint_mesh"')

    # RMG's own 'kpoint_distribution' input keyword sets pct.pe_kpoint directly
    # (confirmed in the RMG source: Input/ReadDynamics.cpp registers it against
    # &pct.pe_kpoint; Misc/InitPe4kpspin.cpp uses it as exactly that -- how
    # many-fold to replicate the processor grid across k-point-parallel
    # groups, with each rank looping over the remaining k-points sequentially
    # within its group). It is NOT the total number of k-points -- an earlier
    # version of this function wrote the total k-point count (product of
    # kpoint_mesh) under this same key, which would have silently
    # misconfigured pe_kpoint (or been rejected by RMG's own validation and
    # fallen back to auto-factorization) rather than doing what was intended.
    #
    # kpoint_multiplier is this codebase's own name for the same pe_kpoint
    # concept, kept distinct so it's never confused with kpoint_mesh's total
    # count, and poppable from input_args like the other numeric knobs above.
    # Default of 1 means no k-point parallelization at all: a single
    # processor grid replica loops over every k-point sequentially, and
    # total_nodes below reflects the processor grid alone.
    kpoint_multiplier = input_args.pop('kpoint_multiplier', kpoint_multiplier)
    input_args['kpoint_distribution'] = kpoint_multiplier

    if 'pseudo_dir' in input_args:
        pseudopotentials_directory = input_args['pseudo_dir']
        if 'pseudopotential' in input_args:
            pseudo_dct = _parse_map(input_args['pseudopotential'])

    total_electrons = _sum_electrons(structure, pseudopotentials_directory, pseudo_dct)

    if 'unoccupied_fraction' in input_args:
        electronic_states = np.ceil(0.5 * total_electrons)
        input_args['unoccupied_states_per_kpoint'] = int(input_args['unoccupied_fraction'] * electronic_states)
        input_args.pop('unoccupied_fraction', 0)

    if 'per_atom_energy' in input_args:
        energy_convergence_criterion = input_args['per_atom_energy'] * len(structure)
        if energy_convergence_criterion < 1e-20:
            energy_convergence_criterion = 1e-20
        elif energy_convergence_criterion > 1e-07:
            energy_convergence_criterion = 1e-07
        else:
            pass
        input_args['energy_convergence_criterion'] = _round_sig(energy_convergence_criterion)
        input_args.pop('per_atom_energy', 0)

    if 'per_atom_rms' in input_args:
        rms_convergence_criterion = input_args['per_atom_rms'] * len(structure)
        if rms_convergence_criterion > 1e-03:
            rms_convergence_criterion = 1e-03
        input_args['rms_convergence_criterion'] = _round_sig(rms_convergence_criterion)
        input_args.pop('per_atom_rms', 0)

    if 'processor_grid' not in input_args:
        fix_nodes = True
        if not target_nodes:
            target_nodes = (total_electrons / (electrons_per_gpu * gpus_per_node))
            fix_nodes = False
        processor_grid, target_nodes = get_processor_grid(
            [int(g) for g in wavefunction_grid.split()],
            target_nodes, gpus_per_node, kpoint_multiplier,
            grid_divisibility_exponent, fix_nodes
        )
        input_args['processor_grid'] = processor_grid

    return input_args, target_nodes


class RMGInput:
    def __init__(self, structure: Structure = None, site_params: dict = None, keywords: dict = None, input_file: str = None, target_nodes: int = 0):
        """
        Initialize the RMGInput class.

        Parameters:
        - structure (pymatgen.core.Structure): Structure object defining the system.
        - keywords (dict): Dictionary of settings (likely from a .yml file).
        - input_file (str): Path to an existing rmg_input file (if reading from a file).
        """
        self.target_nodes = target_nodes

        if input_file:
            # Load from an existing file
            self._load_from_file(input_file)
        elif structure is not None and keywords is not None and site_params is not None:
            # Initialize from a structure and a dictionary of settings
            self.structure = structure
            self.keywords = keywords
            self.site_params = site_params
        else:
            raise ValueError("Must provide either input_file or (structure and keywords and site_params).")

    def _load_from_file(self, input_file: str):
        """Loads an existing RMG input file."""
        with open(input_file, "r") as f:
            lines = f.readlines()

        # Process input file contents (this part depends on the RMG input format)
        self.structure, self.site_params, self.keywords = self._parse_rmg_input(lines)

    def _parse_rmg_input(self, lines):
        """
        Parses an RMG input file into a dictionary of settings and extracts structure information.

        Parameters:
        - lines (list of str): Lines from the input file.

        Returns:
        - structure (pymatgen Structure or None): Parsed structure, if available.
        - keywords (dict): Dictionary of input settings.
        """
        keywords = {}
        structure_params = {
            "lattice_vectors": [],
            "atomic_positions": [],
            "atomic_coordinate_type": None,
            "crds_units": None,
            "bravais_lattice_type": None,
        }
        site_params = {
            "selective_dynamics": [],
            "magnetic_properties": [],
        }
        current_key = None  # Tracks ongoing multi-line values
        multiline_buffer = []  # Stores accumulated lines for multi-line values

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith('"'):  # Skip empty lines and comments
                if current_key and multiline_buffer:
                    keywords[current_key] = '\n'.join(multiline_buffer).replace('"', '').strip()
                current_key = None
                multiline_buffer = []
            elif any(line.startswith(k) for k in ["lattice_vector", "atoms"]):
                current_key = line.split("=")[0].strip()  # Get the key (e.g., "lattice_vector")
                multiline_buffer.append(line.split("=")[1])
            elif current_key and multiline_buffer:
                multiline_buffer.append(line)
            else:
                match = re.match(r'(\w+)\s*=\s*"([^"]*)"', line)
                if match:
                    key, value = match.groups()
                    # Store normal key-value pairs
                    keywords[key] = value.rstrip()

                    # Store relevant structure parameters
                    if key in structure_params:
                        structure_params[key] = value.rstrip()

        # Determine if conversion from Bohr to Angstrom is needed
        conversion_factor = 1.0  # Default (Angstrom)
        if structure_params["crds_units"] == "Bohr":
            conversion_factor = BOHR_TO_ANGSTROM

        # Convert lattice vectors if present
        if "lattice_vector" in keywords:
            structure_params["lattice_vectors"] = np.array([
                list(map(float, line.strip('"').split())) for line in keywords["lattice_vector"].split("\n") if line
            ]) * conversion_factor  # Apply unit conversion

        # Convert atomic positions if present
        if "atoms" in keywords:
            structure_params["atomic_positions"] = [
                line.strip('"').split() for line in keywords["atoms"].split("\n") if line
            ]

        # Build pymatgen Structure if possible
        structure = None
        if structure_params["lattice_vectors"] is not [] and structure_params["atomic_positions"] is not []:
            species = [atom[0] for atom in structure_params["atomic_positions"]]
            coords = np.array([[float(x) for x in atom[1:4]] for atom in structure_params["atomic_positions"]])
            site_params['selective_dynamics'] = [[atom[i] == "1" for i in range(4, 7)] for atom in structure_params["atomic_positions"]]
            site_params['magnetic_properties'] = [[float(atom[i]) for i in range(7, len(atom))] for atom in structure_params["atomic_positions"]]
            coords *= conversion_factor  # Apply unit conversion to atomic positions

            # Set coords_are_cartesian
            if structure_params["atomic_coordinate_type"] == "Absolute":
                coords_are_cartesian = True
            else:
                coords_are_cartesian = False

            structure = Structure(
                structure_params["lattice_vectors"], species, coords,
                coords_are_cartesian=coords_are_cartesian
            )

        # Remove the structure-specific keys from the keywords dictionary
        for key in ('atoms', 'lattice_vector', 'bravais_lattice_type', 'crds_units', 'lattice_units', 'atomic_coordinate_type'):
            keywords.pop(key, 0)

        return structure, site_params, keywords

    def save(self, filename: str):
        """Writes the RMG input file from the current structure and settings."""
        with open(filename, "w") as f:
            f.write(self._generate_rmg_input())

    def _generate_rmg_input(self) -> str:
        """
        Converts structure and keywords into the RMG input file format.

        Returns:
        - str: Formatted input file content.
        """
        writelines = ""
        for key in sorted(self.keywords.keys()):
            writelines += f'{key} = "{self.keywords[key]}"\n'
        writelines += '\n'

        writelines += f'atomic_coordinate_type = "Absolute"\n'
        writelines += f'crds_units = "Angstrom"\n'
        writelines += f'lattice_units = "Angstrom"\n\n'

        lattice_vector_lines = 'lattice_vector = "\n'
        for row in self.structure.lattice.matrix:
            lattice_vector_lines += ' '.join(f"{f:.12e}" for f in row) + '\n'
        lattice_vector_lines += '"\n'
        writelines += lattice_vector_lines

        writelines += f'atoms = "\n'
        for i, site in enumerate(self.structure):
            atom_line = ''
            atom_line += str(site.specie) + ' '
            atom_line += " ".join(f"{val:.12e}" for val in site.coords) + ' '
            atom_line += self.site_params['selective_dynamics'][i] + ' '
            atom_line += self.site_params['magnetic_properties'][i] + '\n'
            writelines += atom_line
        writelines += '"'

        return writelines

    @classmethod
    def from_yaml(cls, yaml_path, structure_path=None, structure_obj=None, pseudopotentials_directory='',
                  magmom_path=None, target_nodes=0, gpus_per_node=8, electrons_per_gpu=10,
                  grid_divisibility_exponent=3, kpoint_multiplier=1):
        with open(yaml_path, 'r') as f:
            input_args = yaml.safe_load(f)

        if not structure_obj:
            structure_obj = Structure.from_file(structure_path)
        site_params = {'selective_dynamics': cls._read_selective_dynamics(structure_obj),
                       'magnetic_properties': cls._read_magnetic_occupancies(structure_obj)}

        if magmom_path and os.path.exists(magmom_path):
            print(f'Reading magnetic moments from {magmom_path}')
            with open(magmom_path, 'r') as f:
                site_params['magnetic_properties'] = [" ".join(map(str, mag)) for mag in json.load(f)]
        elif site_params['magnetic_properties']:
            pass
        else:
            site_params['magnetic_properties'] = ["0.0 0.0 0.0" for site in structure_obj]

        pseudo_dct = {}
        if 'pseudo_dir' in input_args:
            pseudopotentials_directory = input_args['pseudo_dir']
            if 'pseudopotential' in input_args:
                pseudo_dct = _parse_map(input_args['pseudopotential'])

        input_args, target_nodes = compute_grid_and_resources(
            structure_obj, input_args, target_nodes=target_nodes, gpus_per_node=gpus_per_node,
            electrons_per_gpu=electrons_per_gpu, grid_divisibility_exponent=grid_divisibility_exponent,
            pseudopotentials_directory=pseudopotentials_directory, pseudo_dct=pseudo_dct,
            kpoint_multiplier=kpoint_multiplier
        )

        return cls(structure=structure_obj, keywords=input_args, site_params=site_params, target_nodes=target_nodes)

    @staticmethod
    def _read_selective_dynamics(structure):
        return [" ".join("1" if x else "0" for x in sd) if "selective_dynamics" in structure.site_properties else "1 1 1"
            for sd in structure.site_properties.get("selective_dynamics", [[True, True, True]] * len(structure))]

    @staticmethod
    def _read_magnetic_occupancies(structure):
        return [" ".join(str(x) for x in sd) if "magnetic_properties" in structure.site_properties else "0.0 0.0 0.0"
            for sd in structure.site_properties.get("magnetic_properties", [])]
