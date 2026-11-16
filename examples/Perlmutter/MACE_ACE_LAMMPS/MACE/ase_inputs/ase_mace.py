"""
MACE relaxation + single-point driver. Consolidates the companion Pd-Se
pipeline's own two separate files (MACE/relaxation/training/mace_ase_inputs/
ase_mace.py's run_ase_relaxation and MACE/single_points/mace_ase_inputs/
ase_mace.py's run_ase_single_points) into one -- they share the same
MACE-model-loading boilerplate and only genuinely differ in what each does
with the loaded calculator, so there's no reason for two files. Each
function's own real logic is otherwise unchanged.

Dispatched by MatEnsemble via EnsembleFFFit.base.MDMatEnsemble.run_individual
-- a plain entry function taking parallel lists of inputs, runnable
standalone via
`python ase_mace.py run_ase_single_points '["ff1", ...]' '["struct1", ...]' '["output1", ...]' '["cfg1.json", ...]'`.
"""
import json
import os
import sys

import torch
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import MACECalculator

from EnsembleFFFit.utilities.general import parse_list


def _properties(atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return {
        'energy': float(energy),
        'fx': [float(f[0]) for f in forces],
        'fy': [float(f[1]) for f in forces],
        'fz': [float(f[2]) for f in forces],
    }


def _trajectory_id(output):
    """
    Everything in `output`'s own path above its frame-index leaf -- the
    same "original MD trajectory" concept downselect_structures.py groups
    by (see its own docstring), computed once here at write time instead
    of re-derived from path parsing every time it's needed downstream.
    Best-effort: if `output` doesn't have at least 2 path segments, the
    whole path is used as-is rather than raising (this is metadata for
    optional downstream grouping, not something that should fail a chore).
    """
    parts = os.path.normpath(output).split(os.sep)
    return os.sep.join(parts[:-1]) if len(parts) > 1 else output


def _path_derived_metadata(output, structures_root, phase_candidates):
    """
    Resolves 'phase' by checking which of `phase_candidates` (the SAME
    phase_label strings the structure generator for this tier was itself
    configured with, e.g. ["Pd-Se/PdSe", "Pd-Se/Pd2Se3", "Pd-Se/Pd3Se10"])
    appears as a CONTIGUOUS subpath of `output`'s own path relative to
    structures_root, taking that candidate's own LAST path segment (e.g.
    "PdSe", not the full "Pd-Se/PdSe") as the phase value.

    Exists because tier 3/4's defect and motif generators write structures
    under output_root/<phase_label>/... (one batch call spans MULTIPLE
    phases at once -- see generate_defect_structures.py/
    generate_motif_structures.py's own docstrings), so a single static
    metadata value (the `metadata` dict below, constant for the whole
    batch) can't capture which phase a given structure actually is. Static
    per-batch config already covers 'tier' (and would cover 'md_type'/
    'temperature' for tiers where those truly are constant per batch);
    'phase' specifically needs to vary PER STRUCTURE within one batch, so
    it's derived from path instead -- the SAME already-established pattern
    _trajectory_id uses above, not a new convention.

    Matching against known candidate STRINGS, not a fixed path INDEX
    (an earlier version of this function did that, and was wrong): tier 3
    and tier 4 nest the phase label at DIFFERENT depths under
    structures_root -- tier 4's defect/motif axis adds a leading
    "defect"/"motif_<T>k" path segment tier 3 doesn't have (confirmed by
    reading workflow_config.yaml's own run_md_tier4_defect/
    run_md_tier4_motif_300k output_root values) -- and phase_label strings
    can themselves contain a "/" (the validation set's own
    <chemsys>/<phase> convention, e.g. "Pd-Se/PdSe", confirmed from
    generate_structures_tier4_defect's own phase_labels config). A single
    fixed index would need to differ per tier and would silently break if
    either tier's own directory depth ever changes; matching the known
    candidate string instead is robust to both.

    Returns {} if structures_root or phase_candidates is falsy (tiers 1/2,
    which have no phase axis at all). Raises ValueError if phase_candidates
    is given but none of them actually matches -- a silent {} here would
    leave 'phase' missing without any signal that something's misconfigured.
    """
    if not structures_root or not phase_candidates:
        return {}
    rel_parts = os.path.normpath(os.path.relpath(output, structures_root)).split(os.sep)
    for candidate in phase_candidates:
        candidate_parts = os.path.normpath(candidate).split(os.sep)
        n = len(candidate_parts)
        for i in range(len(rel_parts) - n + 1):
            if rel_parts[i:i + n] == candidate_parts:
                return {'phase': candidate_parts[-1]}
    raise ValueError(
        f"None of phase_candidates={phase_candidates} matched any contiguous subpath of "
        f"{os.sep.join(rel_parts)!r} ({output!r} relative to structures_root={structures_root!r}) -- "
        f"check that phase_candidates matches the same phase_labels the structure generator for "
        f"this tier was actually configured with.")


def _write_metadata(output, atoms, static_metadata, structures_root=None, phase_candidates=None):
    """
    Writes metadata.json alongside properties.json -- {tier, phase,
    strain, defect_concentration, md_type, temperature} (whichever are
    present in static_metadata, i.e. known from this stage's own config,
    constant across the whole batch, OR resolved per-structure from path
    via phase_candidates -- see _path_derived_metadata's own docstring for
    why 'phase' specifically needs this) plus two values computed per
    structure: 'stoichiometry' (the reduced chemical formula) and
    'trajectory_id' (see _trajectory_id above). Consumed by
    downselect_structures.py's own group_by and, later, the ACE-fitting
    stage's training_subsets selection -- see run_pipeline.py's own
    run_single_points docstring. Path-derived fields are resolved AFTER
    static_metadata so a path-derived value always wins on a key
    collision (shouldn't happen in practice -- config shouldn't set both
    for the same key -- but per-structure truth should win over a stale
    static default if it ever does).
    """
    metadata = dict(static_metadata or {})
    metadata.update(_path_derived_metadata(output, structures_root, phase_candidates))
    metadata['stoichiometry'] = atoms.get_chemical_formula(mode='hill', empirical=True)
    metadata['trajectory_id'] = _trajectory_id(output)
    with open(os.path.join(output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def run_ase_relaxation(ff_list, struct_list, output_list, in_file_list=None):
    """
    For each (force field, structure, output) triple: load the MACE model
    and the (already rattled, per generate_training_tier1.py) starting
    structure, record its pre-relaxation single point as
    rattled_properties.json -- genuine off-equilibrium forces, kept
    alongside the relaxed point since a training set built only from
    relaxed/near-zero-force configurations gives the ACE fit little to
    learn restoring forces from -- then minimizes cell+positions via
    FIRE/FrechetCellFilter and records the converged single point as
    properties.json. No separate single-point stage runs afterward: the
    optimizer already evaluates the MACE calculator at convergence, so a
    second recompute would just repeat that same work.

    in_file_list entries (optional, same per-task json-config convention as
    ase_mace_md.py's own in_file) may set 'fmax'/'optsteps'/'metadata';
    defaults below are used for any task whose in_file is None or missing
    those keys.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list)), "All lists must be same length"
    if in_file_list is None:
        in_file_list = [None] * n

    for ff, struct, output, in_file in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        cfg = {}
        if in_file:
            with open(in_file) as f:
                cfg = json.load(f)
        fmax = cfg.get('fmax', 0.05)
        optsteps = cfg.get('optsteps', 200)

        # enable_cueq, not enable_oeq -- see run_ase_single_points' own
        # comment below for why.
        calculator = MACECalculator(model_path=ff, enable_cueq=True,
                                    device="cuda" if torch.cuda.is_available() else "cpu")
        atoms = read(struct)
        atoms.set_calculator(calculator)

        # Pre-relaxation (rattled) single point -- genuine off-equilibrium
        # forces, see this function's own docstring for why this is kept.
        with open(os.path.join(output, 'rattled_properties.json'), "w") as f:
            json.dump(_properties(atoms), f, indent=4)

        # Relax (cell + positions) and record the converged single point.
        opt = FIRE(FrechetCellFilter(atoms))
        opt.run(fmax=fmax, steps=optsteps)

        write(filename=os.path.join(output, 'POSCAR'), images=atoms)
        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(_properties(atoms), f, indent=4)
        _write_metadata(output, atoms, cfg.get('metadata'),
                         structures_root=cfg.get('structures_root'),
                         phase_candidates=cfg.get('phase_candidates'))

    return {"status": "complete"}


def run_ase_single_points(ff_list, struct_list, output_list, in_file_list=None):
    """
    For each (force field, structure, output) triple, load the MACE model
    and structure, run a single-point energy/force calculation, and write
    the structure (as POSCAR), computed properties (properties.json), and
    a metadata.json sidecar into `output`.

    in_file_list entries (optional, same per-task json-config convention as
    run_ase_relaxation above) may set 'metadata' -- a dict of whichever of
    {tier, phase, strain, defect_concentration, md_type, temperature} are
    known/constant for this whole batch (see run_pipeline.py's own
    run_single_points docstring for the full field list and how they get
    there from config) -- and, for tiers where a field like 'phase' varies
    PER STRUCTURE within one batch, 'structures_root' + 'phase_candidates'
    (see _path_derived_metadata's own docstring for why and how). Unlike the
    companion pipeline's own version of this function (which left
    in_file_list unused -- plain single points don't need an MD-style
    recipe file), this version DOES read it, purely to carry metadata
    through.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list)), "All lists must be same length"
    if in_file_list is None:
        in_file_list = [None] * n

    for ff, struct, output, in_file in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        cfg = {}
        if in_file:
            with open(in_file) as f:
                cfg = json.load(f)

        # Load the MACE model. enable_cueq (cuequivariance-accelerated
        # tensor products), not enable_oeq -- MACE has two separate
        # acceleration backends gated on two different packages:
        # enable_oeq needs `openequivariance` (the ROCm-side extra, see
        # Ensemble-FF-Fit/pyproject.toml), enable_cueq needs
        # `cuequivariance`/`cuequivariance_torch` (the CUDA extra, what
        # this example's own Dockerfile.ace installs) -- enable_oeq fails
        # outright on a CUDA image where only the cuequivariance extra is
        # installed, so this is a real choice, not an arbitrary default.
        calculator = MACECalculator(model_path=ff, enable_cueq=True,
                                    device="cuda" if torch.cuda.is_available() else "cpu")

        atoms = read(struct)
        write(filename=os.path.join(output, 'POSCAR'), images=atoms)

        atoms.set_calculator(calculator)
        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(_properties(atoms), f, indent=4)
        _write_metadata(output, atoms, cfg.get('metadata'),
                         structures_root=cfg.get('structures_root'),
                         phase_candidates=cfg.get('phase_candidates'))

    return {"status": "complete"}


_ENTRY_POINTS = {
    'run_ase_relaxation': run_ase_relaxation,
    'run_ase_single_points': run_ase_single_points,
}

if __name__ == "__main__":
    _ENTRY_POINTS[sys.argv[1]](
        parse_list(sys.argv[2]), parse_list(sys.argv[3]), parse_list(sys.argv[4]),
        parse_list(sys.argv[5]) if len(sys.argv) > 5 else None)
