"""
Convert a pymatgen-readable structure file (e.g. POSCAR) into a LAMMPS data
file (structure.lmp), for use as the `structure` MDMatEnsemble slot in
ReaxFF LAMMPS runs. Generic: element/atom-type ordering is derived from the
structure's own composition (sorted by atomic number), not hardcoded to any
one example's elements -- so this works for whatever structure is passed
in, not just a specific reference case (per Perlmutter_Pipeline_Wiring.md's
explicit requirement).

Deliberately minimal: unlike the reference example
(run_pipeline/logic_locations/LAMMPs/inputs_directory/Bi2Se3_layered/
{POSCAR,structure.lmp}), which applies a supercell-replication step when
generating structure.lmp (confirmed: its structure.lmp has 360 atoms
against a 15-atom POSCAR, a supercell multiple of it), no such replication
is done here -- Perlmutter_Pipeline_Wiring.md explicitly scopes that ("bells
and whistles") as follow-up work, not something to block the core
conversion logic on. Callers needing a supercell should build one on the
pymatgen Structure before calling structure_to_lmp directly (e.g. via
pymatgen.transformations.advanced_transformations.
CubicSupercellTransformation, as run_pipeline.py's sample_ft_md_structures
stage now does for the finite-temperature coordination-stability check),
not rely on this module to do it -- poscar_to_structure_lmp itself stays a
thin Structure.from_file + structure_to_lmp wrapper, no supercell logic.

UNVERIFIED: pymatgen.io.lammps.data.LammpsData.from_structure's exact
keyword-argument shape (ff_elements=, atom_style=) is inferred from general
familiarity with pymatgen's LAMMPS I/O module, not confirmed against a real
import in this environment (pymatgen isn't installed outside the
MatEnsemble/EnsembleFFFit container, and Perlmutter's compute nodes were
degraded for the entirety of this pipeline's development). If this errors
on the actual container, check `LammpsData.from_structure`'s real signature
there first (e.g. via `python -c "import inspect; from
pymatgen.io.lammps.data import LammpsData;
print(inspect.signature(LammpsData.from_structure))"`) before assuming
something else is wrong.
"""
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData


def structure_to_lmp(structure, output_path, atom_style='charge'):
    """
    Write a pymatgen Structure directly to a LAMMPS data file at
    `output_path`. Factored out of poscar_to_structure_lmp so callers that
    need to transform the Structure first (e.g. building a supercell via
    pymatgen's CubicSupercellTransformation, per this module's own
    docstring above) can do so on the Structure object itself and then
    write it out, without a redundant POSCAR round-trip.

    Returns the ordered list of element symbols (atom-type 1, 2, 3, ...)
    so callers can cross-check against
    EnsembleFFFit.molecular_dynamics.helpers.get_elements(output_path) if
    needed -- both derive the same ordering (sorted by atomic number) from
    the same structure, so they should always agree.
    """
    # Atom-type ordering: unique elements present, sorted by atomic number --
    # deterministic and generic, not tied to any one structure's own
    # element-listing order in its source file.
    ff_elements = sorted(
        {str(sp) for sp in structure.composition.elements},
        key=lambda el: Element(el).Z,
    )

    lammps_data = LammpsData.from_structure(structure, ff_elements=ff_elements, atom_style=atom_style)
    lammps_data.write_file(output_path)
    return ff_elements


def poscar_to_structure_lmp(poscar_path, output_path, atom_style='charge'):
    """
    Read `poscar_path` (any pymatgen-readable structure file, not just a
    file literally named POSCAR), write a LAMMPS data file to
    `output_path`. See structure_to_lmp for the shared conversion logic
    and its return value.
    """
    structure = Structure.from_file(poscar_path)
    return structure_to_lmp(structure, output_path, atom_style=atom_style)
