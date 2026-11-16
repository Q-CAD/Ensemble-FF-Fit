"""
Generates a third ("training_3") finite-temperature MD training tier, seeded
from deliberately defect-substituted (Pd<->Se site-swapped) supercells of the
real validation phases themselves -- NOT the validation structures directly
(that would leak validation into training), but at defect concentrations
(5%/15%/30% of sites) far beyond anything physically realistic, specifically
so the resulting training structures stay structurally distinct from any
held-out validation frame while still covering the SAME chemical motifs
(the same composition's own crystal structure, near-equilibrium bonding)
that validation's own near-hull phases need -- see chat history for the
full reasoning (why training_2's broader near-equilibrium coverage wasn't
specific enough, and why a handful of low-defect-density single-site
substitutions were rejected as too close to validation's own exact
structures -- "a single site-swap in a large supercell").

Standalone, swappable driver script -- same import_module_from_path
convention as downselect_low_energy.py/generate_pyxtal_structures.

Starts from each validation phase's own PRIMITIVE cell, not its existing
(already MACE-relaxed, already supercell-sized) MACE/finite_temperature/
structures/validation POSCAR directly -- CONFIRMED (2026-10-14) that POSCAR
is no longer the as-downloaded Materials Project structure (relax_structures
writes its own relaxed geometry back IN PLACE, overwriting whatever raw
mp_query result was there originally), so pymatgen's own
SpacegroupAnalyzer.find_primitive() is used to recover a genuine primitive
cell FROM that already-relaxed geometry instead -- self-consistent with
whatever this pipeline already treats as "relaxed" for this composition,
with no risk of a lattice-parameter mismatch against a fresh Materials
Project query (see chat history for why re-querying was considered and
rejected).

Supercell construction reuses balanced_supercell's own aspect-ratio-
minimizing multiplicity search (ported from pyxtal_generate.py/
ase_mace_md.py: picks the (nx,ny,nz) repeat that minimizes max/min
lattice-length ratio), but INVERTED to a MINIMUM-atoms floor rather than a
maximum-atoms ceiling -- training_3's own supercells need to be LARGER than
any validation image (max 96 atoms across all 11 phases, confirmed directly
against the real exported reference data), both so a defect-substituted
cell is never literally the same size/shape as a validation image (further
reducing leakage risk, on top of the defect substitution itself) and so the
concentration grid (5%/15%/30%) has enough atoms for reasonable rounding
precision.

Defect generator is a direct, uniform-random pairwise Pd<->Se SITE SWAP
(exchange which element occupies two sites, exactly preserving the
structure's own stoichiometry) -- deliberately NOT a pymatgen Transformation
(PartialRemoveSitesTransformation/SubstitutionTransformation and similar
enumerate symmetry-distinct configurations internally, confirmed slow at
the supercell sizes/concentration grid this needs -- see chat history) --
a plain random-sample-based swap is adequate here since this is
deliberately exploratory, unphysically-high-concentration data, not a
careful physical defect-energetics study.
"""
import os
import random
from itertools import product
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def rescale_volume(structure, volume_fraction):
    """
    Isotropic cell rescale -- same as downselect_low_energy.py's own
    rescale_volume (duplicated, not imported, matching this pipeline's
    convention of each generate_structures driver being self-contained).
    """
    scaling = volume_fraction ** (1 / 3)
    new_matrix = structure.lattice.matrix * scaling
    return Structure(lattice=new_matrix, species=structure.species, coords=structure.frac_coords,
                      coords_are_cartesian=False)


def balanced_supercell_min_atoms(structure, min_atoms, max_mult=6, ceiling_factor=2.0, score_std=False):
    """
    Inverted form of ase_mace_md.py's own balanced_supercell (a min-atoms
    FLOOR instead of a max-atoms CEILING) -- training_3's supercells need
    to be LARGER than any validation image, not capped, so that existing
    function can't be reused as-is. Same aspect-ratio-minimizing search
    (max/min lattice-length ratio, or std if score_std) over every
    (nx, ny, nz) in [1, max_mult]^3, restricted to multiplicities producing
    at least min_atoms sites -- further capped at min_atoms * ceiling_factor
    so a marginally-better aspect ratio at a much larger multiplicity
    doesn't produce a needlessly expensive supercell for later MD. Widens
    the cap (doubling) if nothing clears the floor within it, printing a
    note; falls back to the smallest multiplicity that clears the floor at
    all (regardless of aspect ratio) if even an unbounded search within
    max_mult can't reach min_atoms, rather than raising -- a primitive cell
    larger than min_atoms/2 or so would hit this with max_mult=6, which
    isn't expected for this project's own compositions but shouldn't crash
    a whole batch over one phase.

    Returns (supercell, (nx, ny, nz)).
    """
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


def generate_site_swaps(structure, concentration, rng):
    """
    Returns a NEW Structure with `concentration` fraction of sites'
    element identities perturbed via pairwise Pd<->Se swaps (exchanging
    which element occupies two sites, exactly preserving stoichiometry).

    n_swaps = floor(concentration * natoms / 2) -- each swap exchanges one
    Pd site and one Se site, affecting 2 sites total, so this is the
    largest integer number of PAIRS that doesn't exceed the target
    fraction of AFFECTED SITES (not swap events).

    Falls back to one-way substitutions (converting a remaining untouched
    site directly to the other element, no partner) once one species' own
    site pool is GENUINELY exhausted (n_swaps actually capped by
    len(pd_indices)/len(se_indices), not merely by target_affected//2), to
    still approach the target affected-site fraction -- these one-way
    substitutions DO shift the structure's own nominal stoichiometry away
    from the parent phase. CONFIRMED NECESSARY to distinguish from a
    DIFFERENT, harmless trigger (2026-10-14, caught via a real smoke test
    that initially showed "fallback" firing on nearly every phase): pairwise
    swaps always affect an EVEN number of sites (2 per swap), so whenever
    target_affected is ODD, the very last site can never be reached by
    swaps alone regardless of how much of either species remains -- a pure
    rounding artifact, not exhaustion, and not worth flagging or correcting
    with a stoichiometry-shifting one-way substitution for a single site out
    of hundreds. used_fallback is only ever True for genuine exhaustion
    (confirmed, after this fix, not to fire for any of this project's own
    11 validation phases at 5%/15%/30% -- every phase's own minority-species
    fraction exceeds half the largest configured concentration, with
    margin); the odd-target-by-1-site shortfall is left as a small, harmless
    undershoot of actual_concentration vs. the nominal target instead.

    Returns (new_structure, info) where info has {'n_swaps',
    'n_fallback_substitutions', 'n_sites_affected', 'actual_concentration',
    'used_fallback'}.
    """
    natoms = len(structure)
    target_affected = int(round(concentration * natoms))

    pd_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'Pd']
    se_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'Se']
    rng.shuffle(pd_indices)
    rng.shuffle(se_indices)

    n_swaps_target = target_affected // 2
    n_swaps = min(n_swaps_target, len(pd_indices), len(se_indices))
    exhausted = n_swaps < n_swaps_target

    new_structure = structure.copy()
    affected = set()
    for i in range(n_swaps):
        pi, si = pd_indices[i], se_indices[i]
        new_structure[pi] = 'Se'
        new_structure[si] = 'Pd'
        affected.add(pi)
        affected.add(si)

    used_fallback = False
    n_fallback = 0
    if exhausted:
        remaining_target = target_affected - len(affected)
        used_fallback = remaining_target > 0
        untouched = [i for i in range(natoms) if i not in affected]
        rng.shuffle(untouched)
        for i in untouched[:remaining_target]:
            current = new_structure[i].specie.symbol
            new_structure[i] = 'Pd' if current == 'Se' else 'Se'
            affected.add(i)
            n_fallback += 1

    info = {
        'n_swaps': n_swaps,
        'n_fallback_substitutions': n_fallback,
        'n_sites_affected': len(affected),
        'actual_concentration': len(affected) / natoms,
        'used_fallback': used_fallback,
    }
    return new_structure, info


def _find_phase_structures(validation_root, structure_filename, traj_filename, phase_labels=None):
    """
    Discovers each validation phase's own PHASE-level (pre-MD, post-relax)
    structure file -- NOT the per-frame POSCARs single_point_structures
    later extracts (MACE/single_points/structures/validation/.../<frame>/
    POSCAR), which all share the same underlying cell and would otherwise
    be visited 12x redundantly per phase. Distinguished by traj_filename
    (md_run.traj) sitting alongside the phase-level POSCAR but never
    alongside a frame-level one -- confirmed directly against the real
    tree, not assumed.

    phase_labels (ADDED for training_4's phase-specific tier, see that
    stage's own comment in workflow_config.yaml): optional allow-list of
    exact phase_label strings (e.g. "Pd-Se/PdSe/mp-571383") -- when given,
    every OTHER discovered phase is skipped. None (the default, training_3's
    own behavior, unchanged) keeps every phase found under validation_root.

    Returns {phase_label (str, e.g. "Pd-Se/Pd34Se11/mp-568593"): POSCAR path}.
    """
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


def generate_defect_structures(validation_root, output_root,
                                concentrations=(0.05, 0.15, 0.30),
                                n_placements=4,
                                min_supercell_atoms=150,
                                max_mult=6,
                                volume_fractions=(0.95, 1.0, 1.05),
                                structure_filename="POSCAR",
                                traj_filename="md_run.traj",
                                seed=0,
                                phase_labels=None):
    """
    For every validation phase found under validation_root (or, if
    phase_labels is given, only those -- see training_4's own
    generate_defect_structures_training_4 config, which restricts this
    exact function to just PdSe/Pd2Se3/Pd3Se10 rather than all 11 phases):
    recover its primitive cell (SpacegroupAnalyzer.find_primitive() on the
    current, already-relaxed phase-level structure), build a balanced
    supercell of at least min_supercell_atoms sites, then for every
    concentration in `concentrations` write n_placements distinct
    site-swapped variants, each isotropically rescaled at every
    volume_fractions entry, to output_root/<phase_label>/
    <concentration_pct>/<placement_idx>/<volume_pct>/POSCAR.

    Returns {"written": [str, ...], "summary": {phase_label: {...}}}.
    """
    output_root = Path(output_root)
    phases = _find_phase_structures(validation_root, structure_filename, traj_filename,
                                     phase_labels=phase_labels)
    if not phases:
        raise RuntimeError(f"No phase-level structure ({structure_filename} alongside {traj_filename}) "
                            f"found anywhere under {validation_root}"
                            f"{f' matching phase_labels={list(phase_labels)}' if phase_labels else ''}.")
    if phase_labels is not None:
        missing = sorted(set(phase_labels) - set(phases))
        if missing:
            raise RuntimeError(f"generate_defect_structures: requested phase_labels {missing} not found "
                                f"under {validation_root} (found: {sorted(phases)}).")
    print(f"generate_defect_structures: found {len(phases)} validation phase(s) under {validation_root}")

    rng = random.Random(seed)
    written = []
    summary = {}
    for phase_label, poscar_path in sorted(phases.items()):
        structure = Structure.from_file(poscar_path)
        primitive = SpacegroupAnalyzer(structure).find_primitive() or structure
        supercell, mult = balanced_supercell_min_atoms(primitive, min_supercell_atoms, max_mult=max_mult)

        phase_summary = {
            'relaxed_natoms': len(structure),
            'primitive_natoms': len(primitive),
            'supercell_natoms': len(supercell),
            'supercell_mult': mult,
        }
        for concentration in concentrations:
            pct_label = f"{round(concentration * 100):d}pct"
            fallback_placements = 0
            for placement_idx in range(n_placements):
                defected, info = generate_site_swaps(supercell, concentration, rng)
                if info['used_fallback']:
                    fallback_placements += 1
                for vf in volume_fractions:
                    rescaled = rescale_volume(defected, vf)
                    vf_label = f"{round(vf * 100):d}pct"
                    out_dir = output_root / phase_label / pct_label / str(placement_idx) / vf_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rescaled.to(filename=str(out_dir / structure_filename), fmt="poscar")
                    written.append(str(out_dir))
            phase_summary[f'{pct_label}_fallback_placements'] = fallback_placements
        summary[phase_label] = phase_summary

    print(f"generate_defect_structures: wrote {len(written)} structure(s) ({len(phases)} phase(s) x "
          f"{len(concentrations)} concentration(s) x {n_placements} placement(s) x "
          f"{len(volume_fractions)} volume fraction(s)) under {output_root}")
    for phase_label, s in sorted(summary.items()):
        mult_str = "x".join(str(m) for m in s['supercell_mult'])
        fallback_note = ", ".join(
            f"{k.replace('_fallback_placements', '')}: {v} fallback" for k, v in s.items()
            if k.endswith('_fallback_placements') and v > 0
        )
        print(f"  {phase_label}: primitive {s['primitive_natoms']} atoms -> supercell "
              f"{s['supercell_natoms']} atoms ({mult_str}){', ' + fallback_note if fallback_note else ''}")

    return {"written": written, "summary": summary}
