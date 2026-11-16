"""
Isolated-atom reference-structure generator -- one single-atom-in-a-box cell
per element, for the isolated-atom E0 single-points build_ff_inputs/
downselect_structures both already require (EnsembleFFFit.potential.mace.
build_ensemble_inputs.get_isolated_atom_e0s reads
MACE/single_points/structures/isolated_elements directly).

Standalone, swappable driver -- referenced from workflow_config.yaml's
generate_isolated_elements.generation_task and dynamically imported via
EnsembleFFFit.utilities.general.import_module_from_path, same convention as
every other generate_structures_* driver.

`elements` is deliberately a config value, not derived by scanning
MACE/structures/{training,validation} or re-parsing generate_validation_
structures' own mp_query.chemsys -- this project's actual element set
already has one authoritative home, build_ff_inputs.elements (every ACE fit
needs it directly, for E0 ordering), so generate_isolated_elements.elements
is a YAML alias to that SAME anchor in workflow_config.yaml rather than a
second place to keep in sync, and rather than a tree-wide scan whose cost
grows with every tier added. See workflow_config.yaml's own
&target_elements anchor.

Box size/placement (20 Angstrom cubic cell, atom at its center) matches the
companion Pd-Se pipeline's own isolated_elements convention exactly (checked
directly against its real Pd/Pd and Se/Se POSCARs) -- large enough that
periodic images don't interact at any MLIP's cutoff radius used in this
project, not a value picked fresh here.
"""
import os

from ase import Atoms
from ase.io import write

BOX_LENGTH = 20.0


def generate_isolated_elements(elements, output_root, box_length=BOX_LENGTH):
    """
    One output_root/<element>/POSCAR per entry in `elements` -- a single
    atom centered in a box_length-Angstrom cubic periodic cell. Resumable
    (skips an element whose POSCAR already exists), matching every other
    generate_structures_* driver's own convention. Returns
    {"n_success": int, "n_skipped": int}; "skipped" here only ever means
    "already present from a prior run" (there's no failure mode otherwise --
    unlike pyxtal/defect/motif generation, nothing here can be physically
    unrealizable).
    """
    os.makedirs(output_root, exist_ok=True)

    n_success, n_skipped = 0, 0
    for element in elements:
        out_dir = os.path.join(output_root, element)
        poscar_path = os.path.join(out_dir, "POSCAR")
        if os.path.exists(poscar_path):
            n_skipped += 1
            continue

        os.makedirs(out_dir, exist_ok=True)
        atoms = Atoms(
            symbols=[element],
            positions=[[box_length / 2, box_length / 2, box_length / 2]],
            cell=[box_length, box_length, box_length],
            pbc=True,
        )
        write(poscar_path, atoms)
        n_success += 1

    print(f"generate_isolated_elements: wrote {n_success} structure(s) "
          f"({n_skipped} already present from a prior run)")
    return {"n_success": n_success, "n_skipped": n_skipped}
