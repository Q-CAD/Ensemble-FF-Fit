import os
import json
from pymatgen.core import Structure
from EnsembleFFFit.density_functional_theory.rmg.rmg_input import RMGInput
from EnsembleFFFit.density_functional_theory.rmg.rmg_log import RMGLog


def pick_best_structure(working_directory, structure_filename='POSCAR', rmg_name='rmg_input', magmom_name='MAGMOM.json'):
    """
    Picks the authoritative structure for an RMG run out of whatever is
    actually present in working_directory, in the same priority order as
    pyRMG's generate_pyrmg_cli.py used: the latest structure out of any
    rmg_input.*.log (a run already happened and moved the atoms, e.g. under
    "Relax Structure"), else the structure embedded in an existing rmg_name
    file (a run happened but produced no usable log, or this is a
    continuation whose rmg_input hasn't been regenerated yet), else a static
    structure_filename (first run, nothing has executed yet).

    This is deliberately called at execution time (inside the driver script,
    right before a calculation actually runs) rather than at task-construction
    time -- build_dft_dcts only ever deals with file paths, since carrying a
    resolved Atoms/Structure through task construction would fix it to
    whatever was on disk when the task list was built, not whatever is
    actually there when the task runs.

    Returns (structure, source) where source is a short string identifying
    which of the above the structure came from, for logging purposes.
    """
    rmg_input_path = os.path.join(working_directory, rmg_name)
    structure_path = os.path.join(working_directory, structure_filename)
    magmom_path = os.path.join(working_directory, magmom_name)

    existing_rmg_input = RMGInput(input_file=rmg_input_path) if os.path.exists(rmg_input_path) else None

    rmg_logs = RMGLog(working_directory)
    structure = None
    source = None

    for log_file in sorted(rmg_logs.logs_data.keys(), reverse=True):
        structures = rmg_logs.logs_data[log_file].get('structures', [])
        if structures:
            structure = structures[-1]
            source = f'final structure of {log_file}'
            break

    if structure is None and existing_rmg_input is not None:
        structure = existing_rmg_input.structure
        source = rmg_name

    if structure is None and os.path.exists(structure_path):
        structure = Structure.from_file(structure_path)
        source = structure_filename

    if structure is None:
        raise FileNotFoundError(
            f"No usable structure found in {working_directory}: checked "
            f"'{rmg_name}.*.log' logs, '{rmg_name}', and '{structure_filename}'."
        )

    if existing_rmg_input is not None:
        for prop_key, prop_value in existing_rmg_input.site_params.items():
            structure.add_site_property(prop_key, prop_value)

    if os.path.exists(magmom_path):
        with open(magmom_path, 'r') as f:
            magmoms = json.load(f)
        structure.add_site_property('magnetic_properties', [" ".join(map(str, m)) for m in magmoms])

    return structure, source
