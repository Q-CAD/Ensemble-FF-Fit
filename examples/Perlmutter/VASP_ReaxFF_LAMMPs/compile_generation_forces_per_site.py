"""
Consolidates per-site, per-component ReaxFF-vs-DFT force comparisons for
the SELECTED force field from EVERY fitting generation found on disk
(generation_0, the unfit seed, through whichever generation_N is latest
-- auto-discovered via compile_generation_energies_per_atom.
_discover_generation_winners, not hardcoded) into two long/tidy tables
(training, validation) -- one row per (structure, atom, cartesian
component) -- ready for scatter plots tracking per-component (fx/fy/fz
separately) or pooled/"net" (fx+fy+fz rows together) agreement as the
fitting progressed across generations.

IMPORTANT (2026-10, see bug_report.md): pymatgen's LammpsData.from_
structure (used to write every reaxff_validation structure.lmp) silently
rotates atomic positions -- and therefore every LAMMPS-computed force --
into LAMMPS's own canonical box frame whenever a structure's own lattice
isn't already in that orientation. DFT's reference forces, read straight
out of vasprun.xml/properties.json, are in VASP's native (unrotated)
frame, so a naive per-component (fx/fy/fz) comparison silently mixes two
different coordinate frames for any such structure. This script corrects
for that the SAME way run_rank_reaxff_validation now does (see
EnsembleFFFit.analysis.best_force_field.rotate_reference_forces_to_
lammps_frame) -- by rotating the parsed DFT reference forces into
LAMMPS's frame immediately after parsing, via the identical rotation
pymatgen itself computes (lattice_2_lmpbox + SymmOp) -- rather than via
any separate post-processing step. (An earlier version of this pipeline
applied this fix retroactively, as a one-off post-processing pass over
an already-compiled CSV -- see correct_lammps_frame_forces.py in
run_pipeline/ for that historical approach; it's superseded by this
in-loop fix and should not be re-run against output from this version of
the script, since the rotation would then be applied twice.)

Why this needs its own script rather than reusing force_deviations.csv/
force_deviations_per_structure_{training,validation}.csv (already written
per generation by write_generation_diagnostics, run_pipeline.py): those
are already REDUCED to one RMSE-per-structure-per-axis number (see
EnsembleFFFit.analysis.best_force_field.get_ff_deviations/
write_per_structure_force_deviations) -- exactly what rank_force_fields_
combined's own objective function needs, but not what a scatter plot
needs. A scatter plot needs the raw, unreduced per-atom fx/fy/fz pairs
(predicted vs DFT) that RMSE was computed FROM -- these were never
themselves written to disk anywhere; they only ever existed transiently
as python arrays inside get_ff_deviations, read straight out of each
structure's own properties.json. So this script re-parses properties.json
directly (EnsembleFFFit.analysis.dict_parsers.PropertiesOnlyParser +
parse_reference_tree, the same machinery run_rank_reaxff_validation uses),
but for exactly ONE ff_label (the generation winner) at a time rather than
the whole candidate ensemble -- avoids re-parsing the entire
(174-candidate-plus) force_fields tree per generation just to pull one
label back out of it.

Where each generation's winner's raw properties.json tree actually lives
-- NOT necessarily under that generation's own MD/generation_N/, same
caveat as write_generation_diagnostics' own docstring (a given seed is
always EVALUATED one generation before it becomes a seed) -- is read
directly off that generation's own FF/generation_N/selection_report.txt
(see _single_point_root_for_generation), NOT guessed as "generation
N-1's root": confirmed directly against this pipeline's own run that a
winner's ff_label is NOT guaranteed globally unique across generations
(coincidental reaxff_inputs sample-numbering reuse), so an existence- or
index-based guess can silently resolve to the WRONG generation's
candidate pool. generation_0 (the unfit seed, 1_10199, never itself
"selected" -- no selection_report.txt of its own) is the one hardcoded
exception; see _single_point_root_for_generation's own docstring.

The FF side's own "run" key comes out one path segment longer than the
DFT reference side's (MDMatEnsemble's run_directory/inputs_directory split
embeds an extra "structures" segment -- see run_pipeline.py's own
_drop_first_run_segment docstring for the full explanation); reused
directly from run_pipeline.py here rather than re-deriving it, so this
stays in lockstep with however run_rank_reaxff_validation itself matches
the two sides up.

Units: kcal/mol/Angstrom throughout (ReaxFF's/LAMMPS's own native "real"
units -- the DFT side is converted from VASP's native eV/Angstrom via the
exact same best_force_field.convert_units_to_kcal_mol used everywhere
else in this pipeline), plus an eV/Angstrom column alongside for
convenience. No divisor/per-atom-unit subtlety here, unlike the energy
compilation (compile_generation_energies_per_atom.py) -- forces are
already a genuinely per-atom, per-component quantity straight out of
properties.json, nothing to normalize by structure size.

Run this from the pipeline's own working directory (same convention as
compile_generation_energies_per_atom.py), inside the matensemble:ff-fit
container (needs pandas + pymatgen; no jax/jaxreaxff import, so CPU-only
is fine):

    python3 compile_generation_forces_per_site.py

Writes compiled_generation_forces_per_site_training.csv and
compiled_generation_forces_per_site_validation.csv in the current
directory.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline import _drop_first_run_segment  # noqa: E402
from compile_generation_energies_per_atom import _discover_generation_winners, _load_reference_roots  # noqa: E402

from EnsembleFFFit.analysis.best_force_field import convert_units_to_kcal_mol, rotate_reference_forces_to_lammps_frame
from EnsembleFFFit.analysis.dict_parsers import PropertiesOnlyParser, parse_reference_tree

EV_TO_KCAL_MOL = 23.060548867

# subset -> that subset's own validation/training subfolder name under a
# single_points/reaxff_validation root (see _find_single_point_root) --
# fixed path-segment conventions used identically everywhere else in
# run_pipeline.py (e.g. its own run_rank_reaxff_validation), not
# user-configurable, so hardcoded here the same way.
SUBSET_SUBFOLDERS = {
    'training': 'training',
    'validation': 'aimd',
}

COMPONENTS = ('fx', 'fy', 'fz')


_COPIED_FROM_RE = re.compile(r'^Copied (.+?)/selected/force_fields/\S+/ffield ->', re.MULTILINE)


def _single_point_root_for_generation(generation):
    """
    The MD/.../single_points/reaxff_validation root whose own rank_
    reaxff_validation run evaluated THIS generation's winner as a
    candidate (one generation before it became a seed) -- read directly
    off FF/<generation>/selection_report.txt's own "Copied ... -> ..."
    line, the exact provenance run_pipeline.py's own
    run_select_next_generation_force_field recorded at selection time.

    Deliberately NOT found by existence-checking a guessed set of
    candidate roots (an earlier version of this script did exactly that)
    -- a winner's own ff_label is NOT guaranteed unique across
    generations: confirmed directly against this pipeline's own run,
    reaxff_run_1_offdiag_section4_sample_002_k1 (generation_2's winner)
    ALSO exists as a plain candidate directory name under generation_1,
    generation_2, AND generation_4's own single_points trees --
    coincidental reaxff_inputs sample-numbering reuse across independent
    generations, not the same evaluation. An existence-based guess can
    silently pick the wrong one; this can't, since it's reading the
    pipeline's own recorded answer.

    generation_0 is the one exception -- the unfit seed, manually placed
    before the pipeline ever ran, never itself "selected" (no
    selection_report.txt of its own) -- hardcoded to the one place it's
    confirmed (filesystem-checked, not assumed) to live.
    """
    if generation == 'generation_0':
        return 'MD/single_points/reaxff_validation'

    report_path = os.path.join('FF', generation, 'selection_report.txt')
    with open(report_path) as f:
        text = f.read()
    m = _COPIED_FROM_RE.search(text)
    if not m:
        raise RuntimeError(f"Couldn't find a 'Copied ... -> ...' provenance line in {report_path}")
    return m.group(1)


def compile_subset(subset, dft_root, generations):
    subfolder = SUBSET_SUBFOLDERS[subset]
    reference_dct = convert_units_to_kcal_mol(
        rotate_reference_forces_to_lammps_frame(
            parse_reference_tree(dft_root, parser_cls=PropertiesOnlyParser), dft_root),
        EV_TO_KCAL_MOL)

    rows = []
    skipped = []
    for generation, ff_label in generations:
        gen_root = _single_point_root_for_generation(generation)
        ff_label_dir = os.path.join(gen_root, subfolder, 'force_fields', ff_label)
        if not os.path.isdir(ff_label_dir):
            raise RuntimeError(f"{generation}/{ff_label}: expected single-point directory "
                               f"{ff_label_dir!r} (derived from {generation}'s own "
                               f"selection_report.txt) does not exist")
        ff_dct = _drop_first_run_segment(
            parse_reference_tree(ff_label_dir, parser_cls=PropertiesOnlyParser))

        for run, images in ff_dct.items():
            if run not in reference_dct:
                skipped.append(f"{generation}/{ff_label}: run {run!r} missing from {dft_root}")
                continue
            for image, pred in images.items():
                if image not in reference_dct[run]:
                    skipped.append(f"{generation}/{ff_label}: image {image!r} missing under run {run!r} in {dft_root}")
                    continue
                ref = reference_dct[run][image]
                structure = f"{run}/{image}" if run else image

                for component in COMPONENTS:
                    pred_vals, dft_vals = pred[component], ref[component]
                    if len(pred_vals) != len(dft_vals):
                        skipped.append(
                            f"{generation}/{ff_label}/{structure}/{component}: atom count mismatch "
                            f"({len(pred_vals)} predicted vs {len(dft_vals)} DFT)")
                        continue
                    for atom_index, (p, d) in enumerate(zip(pred_vals, dft_vals)):
                        rows.append({
                            'generation': generation,
                            'ff_label': ff_label,
                            'structure': structure,
                            'atom_index': atom_index,
                            'component': component,
                            'predicted_kcal_molA': p,
                            'dft_kcal_molA': d,
                            'abs_deviation_kcal_molA': abs(p - d),
                            'predicted_eV_A': p / EV_TO_KCAL_MOL,
                            'dft_eV_A': d / EV_TO_KCAL_MOL,
                        })

    if skipped:
        print(f"[{subset}] skipped {len(skipped)} mismatched run/image/component combo(s), e.g.:")
        for msg in skipped[:5]:
            print(f"    {msg}")

    return pd.DataFrame(rows, columns=[
        'generation', 'ff_label', 'structure', 'atom_index', 'component',
        'predicted_kcal_molA', 'dft_kcal_molA', 'abs_deviation_kcal_molA',
        'predicted_eV_A', 'dft_eV_A'])


if __name__ == '__main__':
    generations = _discover_generation_winners()
    print(f"Discovered {len(generations)} generation(s): "
          + ", ".join(f"{g}={label}" for g, label in generations))
    print()

    for subset, dft_root in _load_reference_roots().items():
        compiled = compile_subset(subset, dft_root, generations)
        out_path = f'compiled_generation_forces_per_site_{subset}.csv'
        compiled.to_csv(out_path, index=False)
        n_structures = compiled.groupby(['generation', 'structure']).ngroups if len(compiled) else 0
        print(f"Wrote {len(compiled)} row(s) ({n_structures} structure(s) across "
              f"{compiled['generation'].nunique()} generation(s)) to {out_path}")
        print()
