"""
Generates training_4's own "motif" axis: isotropically volume-rescaled
(stretched/compressed) supercells of specific phases of interest
(PdSe/Pd2Se3/Pd3Se10 -- see workflow_config.yaml's own
generate_motif_structures_training_4 comment for why these three and not
all 11 validation phases), with NO defect substitution (see
generate_defect_structures.py's own training_3 tier for that, separate
axis). Deliberately a much wider volume-fraction range than training_2/3's
own [0.95, 1.0, 1.05] (structural_stability_screening's own real NPT runs
show these phases excursing far more than +/-5% under genuine MD -- see
chat history), since the whole point of this tier is to cover the volume
range these phases actually explore, not just rattle around equilibrium.

Standalone, self-contained driver script -- same
import_module_from_path/run_pipeline.py dispatch convention as
generate_defect_structures.py/downselect_low_energy.py. _find_phase_structures/
rescale_volume/balanced_supercell_min_atoms are duplicated from
generate_defect_structures.py here (not imported) -- same "each driver
script is self-contained" convention that file's own rescale_volume
docstring documents, so a change to one driver's copy never silently
affects another.

Each (phase, volume_fraction) combination is written out n_seeds times as
byte-identical starting structures, NOT n_seeds distinct perturbations --
this is deliberate, not a missing feature: ase_mace_md.py's own MD driver
(MaxwellBoltzmannDistribution + Langevin) never seeds numpy/Python's global
RNG explicitly (confirmed by reading it -- no np.random.seed/random.seed
anywhere), so n_seeds identical starting structures already yield n_seeds
genuinely different MD trajectories (different initial velocities,
different thermal noise) once run -- no separate rattle/perturbation step
is needed to get independent replicates. The temperature axis itself
(300K/500K/700K/900K) lives in which finite_temperature_md_training_4_*
stage is pointed at this same structures_root, not in anything generated
here -- see those stages' own config comments.
"""
import os
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _find_phase_structures(validation_root, structure_filename, traj_filename, phase_labels=None):
    """Identical to generate_defect_structures.py's own copy -- see that file's docstring for why duplicated."""
    validation_root = Path(validation_root)
    phases = {}
    for root, _, files in os.walk(validation_root):
        root = Path(root)
        if structure_filename in files and (root / traj_filename).exists():
            label = str(root.relative_to(validation_root))
            if phase_labels is not None and label not in phase_labels:
                continue
            phases[label] = root / structure_filename
    return phases


def rescale_volume(structure, volume_fraction):
    """Identical to generate_defect_structures.py's own copy -- see that file's docstring for why duplicated."""
    scaling = volume_fraction ** (1 / 3)
    new_matrix = structure.lattice.matrix * scaling
    return Structure(lattice=new_matrix, species=structure.species, coords=structure.frac_coords,
                      coords_are_cartesian=False)


def balanced_supercell_min_atoms(structure, min_atoms, max_mult=6, ceiling_factor=2.0, score_std=False):
    """Identical to generate_defect_structures.py's own copy -- see that file's docstring for the full reasoning."""
    import numpy as np
    from itertools import product

    lengths = np.array(structure.lattice.abc)
    natoms = len(structure)

    ceiling = min_atoms * ceiling_factor
    best_mult = None
    while best_mult is None and ceiling <= min_atoms * (max_mult ** 3):
        best_score = np.inf
        for nx, ny, nz in product(range(1, max_mult + 1), repeat=3):
            mult_atoms = natoms * nx * ny * nz
            if mult_atoms < min_atoms or mult_atoms > ceiling:
                continue
            new_lengths = lengths * np.array([nx, ny, nz])
            score = np.std(new_lengths) if score_std else max(new_lengths) / min(new_lengths)
            if score < best_score:
                best_score = score
                best_mult = (nx, ny, nz)
        if best_mult is None:
            ceiling *= 2

    if best_mult is None:
        candidates = [(nx, ny, nz) for nx, ny, nz in product(range(1, max_mult + 1), repeat=3)
                      if natoms * nx * ny * nz >= min_atoms]
        if not candidates:
            raise ValueError(f"No supercell multiplicity up to max_mult={max_mult} reaches "
                              f"min_atoms={min_atoms} from a {natoms}-atom primitive cell.")
        best_mult = min(candidates, key=lambda m: natoms * m[0] * m[1] * m[2])
        print(f"balanced_supercell_min_atoms: no aspect-ratio-optimized multiplicity found within "
              f"max_mult={max_mult}; falling back to the smallest multiplicity {best_mult} that "
              f"clears min_atoms={min_atoms} regardless of aspect ratio.")

    supercell = structure.copy()
    supercell.make_supercell(best_mult)
    return supercell, best_mult


def generate_motif_structures(validation_root, output_root, phase_labels,
                               volume_fractions=(0.80, 0.90, 1.00, 1.10, 1.20),
                               n_seeds=3,
                               min_supercell_atoms=150,
                               max_mult=6,
                               structure_filename="POSCAR",
                               traj_filename="md_run.traj"):
    """
    For each phase in phase_labels: recover its primitive cell, build a
    balanced supercell of at least min_supercell_atoms sites (same sizing
    convention as generate_defect_structures.py's own training_3 tier, so
    PdSe/Pd2Se3/Pd3Se10 land at the same already-proven 192/270/208-atom
    cells), then for every volume_fraction write n_seeds byte-identical
    copies to output_root/<phase_label>/<volume_pct>/<seed>/POSCAR (see
    this module's own docstring for why identical copies are correct here,
    not a bug).

    Returns {"written": [str, ...], "summary": {phase_label: {...}}}.
    """
    output_root = Path(output_root)
    phases = _find_phase_structures(validation_root, structure_filename, traj_filename,
                                     phase_labels=phase_labels)
    missing = sorted(set(phase_labels) - set(phases))
    if missing:
        raise RuntimeError(f"generate_motif_structures: requested phase_labels {missing} not found "
                            f"under {validation_root} (found: {sorted(phases)}).")

    written = []
    summary = {}
    for phase_label, poscar_path in sorted(phases.items()):
        structure = Structure.from_file(poscar_path)
        primitive = SpacegroupAnalyzer(structure).find_primitive() or structure
        supercell, mult = balanced_supercell_min_atoms(primitive, min_supercell_atoms, max_mult=max_mult)

        for vf in volume_fractions:
            rescaled = rescale_volume(supercell, vf)
            vf_label = f"{round(vf * 100):d}pct"
            for seed in range(n_seeds):
                out_dir = output_root / phase_label / vf_label / str(seed)
                out_dir.mkdir(parents=True, exist_ok=True)
                rescaled.to(filename=str(out_dir / structure_filename), fmt="poscar")
                written.append(str(out_dir))

        summary[phase_label] = {
            'primitive_natoms': len(primitive),
            'supercell_natoms': len(supercell),
            'supercell_mult': mult,
        }

    print(f"generate_motif_structures: wrote {len(written)} structure(s) ({len(phases)} phase(s) x "
          f"{len(volume_fractions)} volume fraction(s) x {n_seeds} seed(s)) under {output_root}")
    for phase_label, s in sorted(summary.items()):
        mult_str = "x".join(str(m) for m in s['supercell_mult'])
        print(f"  {phase_label}: primitive {s['primitive_natoms']} atoms -> supercell "
              f"{s['supercell_natoms']} atoms ({mult_str})")

    return {"written": written, "summary": summary}
