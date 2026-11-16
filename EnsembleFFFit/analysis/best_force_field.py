from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
import numpy as np
import os

from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure

from EnsembleFFFit.molecular_dynamics.lammps.poscar_to_structure_lmp import lammps_frame_rotation_matrix


def convert_units(run_image_dct, energy_to_ev_factor):
    """
    Returns a NEW {run: {image: props}} dict with "energy"/"fx"/"fy"/"fz"
    divided by energy_to_ev_factor -- e.g. LAMMPS real units (kcal/mol,
    kcal/mol/Angstrom) -> eV/eV per Angstrom. The SAME factor applies to
    forces as to energy, not a separate one: force = energy/distance, and
    the distance unit (Angstrom) is unchanged between LAMMPS real units
    and eV/Angstrom, so only the energy-numerator unit needs converting.
    Does not mutate the input. structure (if present) is passed through
    unchanged.
    """
    out = {}
    for run, images in run_image_dct.items():
        out[run] = {}
        for image, props in images.items():
            converted = dict(props)
            converted['energy'] = props['energy'] / energy_to_ev_factor
            converted['fx'] = [v / energy_to_ev_factor for v in props['fx']]
            converted['fy'] = [v / energy_to_ev_factor for v in props['fy']]
            converted['fz'] = [v / energy_to_ev_factor for v in props['fz']]
            out[run][image] = converted
    return out


def convert_units_to_kcal_mol(run_image_dct, ev_to_kcal_mol_factor):
    """
    Returns a NEW {run: {image: props}} dict with "energy"/"fx"/"fy"/"fz"
    MULTIPLIED by ev_to_kcal_mol_factor -- the opposite direction from
    convert_units above (which divides LAMMPS's native kcal/mol down to
    eV). Used on the DFT side instead: VASP's native eV/eV-per-Angstrom
    output -> kcal/mol/kcal-mol-per-Angstrom, so every energy/force
    comparison in this module stays in ReaxFF's own native unit
    throughout, matching relative_energy_comparison.py's own ReaxEntry-
    normalized values. Reporting in eV (or eV/atom) doesn't work cleanly
    here: get_divisors can solve a different total-atom-count reduction
    for different structures (see RelativeEnergyComparison's own
    docstring), so there's no single, consistent atom count to normalize
    an eV value by across the whole comparison -- kcal/mol sidesteps that
    entirely by just being ReaxFF's native unit, not a per-atom one.
    Does not mutate the input.
    """
    out = {}
    for run, images in run_image_dct.items():
        out[run] = {}
        for image, props in images.items():
            converted = dict(props)
            converted['energy'] = props['energy'] * ev_to_kcal_mol_factor
            converted['fx'] = [v * ev_to_kcal_mol_factor for v in props['fx']]
            converted['fy'] = [v * ev_to_kcal_mol_factor for v in props['fy']]
            converted['fz'] = [v * ev_to_kcal_mol_factor for v in props['fz']]
            out[run][image] = converted
    return out


def rotate_reference_forces_to_lammps_frame(run_image_dct, reference_root):
    """
    Returns a NEW {run: {image: props}} dict with "fx"/"fy"/"fz" rotated
    into LAMMPS's own canonical box frame -- see poscar_to_structure_lmp.
    lammps_frame_rotation_matrix and bug_report.md for the full story:
    pymatgen's LammpsData.from_structure (used anywhere a reaxff_
    validation POSCAR gets converted to a LAMMPS data file) silently
    rotates every atomic position -- and therefore every force LAMMPS
    subsequently computes -- into a canonical box orientation whenever a
    structure's own lattice isn't already in that orientation. DFT's own
    reference forces (read straight from vasprun.xml/properties.json) are
    in VASP's native, unrotated frame, so any comparison that treats
    fx/fy/fz as independent per-axis quantities -- exactly what
    get_ff_deviations/rank_force_fields_combined below do -- silently
    mixes two different coordinate frames unless the reference side is
    rotated to match first. Energy and force MAGNITUDE are rotation-
    invariant and need no correction; this only matters because those
    functions compare components independently.

    Call this on the DFT reference dict BEFORE any force comparison (and
    before/after convert_units_to_kcal_mol -- order doesn't matter, since
    rotation and the scalar unit conversion commute), right after
    parse_reference_tree/parse_labeled_tree produced it. reference_root:
    the same root run_image_dct was parsed from -- every run/image key is
    itself the POSCAR's own subpath under that root (reference_root/
    <run>/<image>/POSCAR always exists wherever that image's
    properties.json does), so no separate structure lookup is needed.
    Rotations are cached per (run, image) within one call, since loading
    a POSCAR + computing lattice_2_lmpbox is the expensive part and a
    structure's own lattice doesn't vary across what it's being compared
    against. Does not mutate the input.
    """
    rotation_cache = {}

    def _rotation_for(run, image):
        key = (run, image)
        if key not in rotation_cache:
            poscar_path = os.path.join(reference_root, run, image, 'POSCAR')
            structure = Structure.from_file(poscar_path)
            rotation_cache[key] = lammps_frame_rotation_matrix(structure)
        return rotation_cache[key]

    out = {}
    for run, images in run_image_dct.items():
        out[run] = {}
        for image, props in images.items():
            op = SymmOp.from_rotation_and_translation(_rotation_for(run, image))
            vecs = np.column_stack([props['fx'], props['fy'], props['fz']])
            rotated = op.operate_multi(vecs)
            converted = dict(props)
            converted['fx'] = rotated[:, 0].tolist()
            converted['fy'] = rotated[:, 1].tolist()
            converted['fz'] = rotated[:, 2].tolist()
            out[run][image] = converted
    return out


def get_relative_energy_deviations(relative_energy_ff_dct, relative_energy_reference_dct, reference_label="DFT"):
    """
    {ff_label: {entry_name: {target: abs(ff_relative_energy -
    dft_relative_energy)}}} -- plain unweighted absolute difference, not
    RMSE, per Paper_Analysis.md's explicit requirement: this is a direct
    training/validation-set accuracy check against the exact relative
    energies ReaxFF was fit against (see
    relative_energy_comparison.RelativeEnergyComparison), not another
    Boltzmann/training-weighted objective function -- those weights
    decide what parse2fit emphasizes during fitting, which has no bearing
    on how far off a fitted potential's prediction actually is.

    Shape-checked the same way get_ff_deviations/rank_force_fields_combined
    check validation/training dicts (assert_comparable_dicts) -- entry_name
    plays the role of md_name, target (a DFT-root-relative path) plays the
    role of md_image.
    """
    assert_comparable_dicts(relative_energy_ff_dct, relative_energy_reference_dct, reference_label)

    ref_root = relative_energy_reference_dct[reference_label]
    deviation_dct = {}

    for ff_label, entry_dct in relative_energy_ff_dct.items():
        deviation_dct[ff_label] = {}
        for entry_name, target_dct in entry_dct.items():
            deviation_dct[ff_label][entry_name] = {}
            for target, props in target_dct.items():
                ref_value = ref_root[entry_name][target]['relative_energy']
                deviation_dct[ff_label][entry_name][target] = abs(props['relative_energy'] - ref_value)

    return deviation_dct


def get_relative_energy_scores(deviation_dct):
    """ Mean absolute relative-energy deviation per force field. """
    score_dct = {}

    for ff_label, entry_dct in deviation_dct.items():
        total, count = 0.0, 0
        for target_dct in entry_dct.values():
            for deviation in target_dct.values():
                total += deviation
                count += 1

        if count == 0:
            raise ValueError(f"No relative-energy comparisons found for force field '{ff_label}'")

        score_dct[ff_label] = total / count

    return score_dct


def rank_force_fields_combined(
    validation_ff_dct, validation_reference_dct,
    training_ff_dct, training_reference_dct,
    validation_force_weight=1.0, training_force_weight=1.0,
    validation_energy_weight=None, training_energy_weight=None, energy_per_atom=False,
    force_aggregation="structure_mean",
    training_relative_energy_dct=None, training_relative_energy_reference_dct=None,
    validation_relative_energy_dct=None, validation_relative_energy_reference_dct=None,
    training_relative_energy_weight=1.0, validation_relative_energy_weight=1.0,
    reference_label="DFT",
):
    """
    Combined ReaxFF validation ranking (see run_pipeline.py's
    run_rank_reaxff_validation docstring for the full pipeline this feeds):

    - validation_ff_dct/validation_reference_dct and training_ff_dct/
      training_reference_dct: {label: {run: {image: props}}} /
      {reference_label: {run: {image: props}}}, from DFT/validation's
      AIMD trajectories and DFT/training respectively -- forces only.
      There used to also be a frame-0-relative validation energy
      component here (see relativize_energies, since removed): it
      compared energies via plain subtraction against each trajectory's
      own frame 0, with no stoichiometric (get_divisors) reduction at
      all -- a genuinely different, less rigorous method than training's
      own relative-energy treatment below, and redundant with it once
      reaxff_validation.yml (see relative_energy_comparison.py) made the
      SAME get_divisors-based method expressible for AIMD trajectories
      too (subtract: [frame 0], get_divisors: True). Removed so training
      and validation now share one consistent energy-comparison method.
    - validation_energy_weight/training_energy_weight (ADDED, optional --
      a DIFFERENT, simpler mechanism than the relative-energy one above,
      for callers whose own energy quantity is already directly
      comparable without stoichiometric normalization, e.g. ACE's
      energy_corrected -- see FF_fitting's own rank_ace_validation
      precedent, which validates against absolute energy_corrected
      directly by design). `None` (the default for BOTH) -- not 0.0 --
      means "omit the energy term and its table column entirely," the
      SAME sentinel convention already used by the relative-energy dcts
      below, so existing callers that never pass these two new params get
      bit-for-bit identical output (verified directly: real synthetic
      before/after comparison against this exact function, both with and
      without relative-energy dcts set, confirmed byte-identical).
      energy_per_atom (default False, see get_ff_deviations' own
      docstring) normalizes the energy RMSE by atom count before
      weighting -- needed for an eV/atom-style convention, irrelevant
      (and ignored) when both energy weights are None.
    - force_aggregation (ADDED, default "structure_mean" -- preserves
      existing behavior exactly): which pooling convention computes
      mean_val_force/mean_train_force. "structure_mean" (the default, via
      get_ff_deviations) takes one RMSE per structure then averages those
      uniformly across structures. "atom_component_pooled" (via the new
      get_pooled_force_rmse) instead pools every force component from
      every atom of every structure into one flat array before a single
      RMSE -- implicitly atom-count-weighted, and the convention pyace/
      tensorpotential's own fitting-time test_metrics.txt uses. These are
      NOT interchangeable (a real, confirmed ~25-35% discrepancy between
      them was found downstream -- see get_pooled_force_rmse's own
      docstring) -- pick whichever one your own fitting-time metric
      actually uses if you want this ranking's numbers to line up with it.
      More conventions can be added here as new elif branches without
      touching the default path, for other MLIP/interatomic-potential
      backends with their own reporting conventions.
    - training_relative_energy_dct/training_relative_energy_reference_dct
      and validation_relative_energy_dct/validation_relative_energy_
      reference_dct (optional): {label: {entry_name: {target:
      {"relative_energy": value}}}}, from
      relative_energy_comparison.build_relative_energy_dcts -- the
      YAML-driven (reaxff_newest_kT.yml/reaxff_validation.yml) relative
      energies, in kcal/mol. Omitted (left None) entirely by default --
      existing callers that don't pass these keep their old behavior and
      table columns unchanged.

    Both ff_dct arguments are assumed ALREADY unit-converted to match
    their reference (kcal/mol/kcal-mol-per-Angstrom throughout -- see
    convert_units_to_kcal_mol) -- this function does no unit conversion
    itself, only RMSE scoring. The relative-energy dcts need no such
    conversion -- RelativeEnergyComparison builds both sides through the
    same ReaxEntry, which already normalizes units via parse2fit's own
    UnitConverter, also kcal/mol.

    Returns (lines, labels, combined_scores): lines is a ranked table
    (best/lowest combined score first) reporting, per force field, each
    raw component (with its own unit in the column header) next to the
    combined weighted score actually used for ranking. labels/
    combined_scores are the same ranking, machine-readable.
    """
    # weight=1.0/1.0 here (not the caller's own weights) -- these calls
    # produce RAW, unweighted per-component RMSE; this function's own
    # validation_force_weight/training_force_weight args are applied
    # afterward, once, when combining components -- same "raw first,
    # weight once at the end" pattern as format_ranking_table's own
    # raw_deviation_dct. energy_per_atom is passed through unconditionally
    # (cheap either way -- get_ff_deviations always computes an energy
    # term as a byproduct of computing forces, whether or not this
    # function goes on to use it) so val_deviation_dct/train_deviation_dct
    # carry the right (normalized-or-not) energy value regardless of
    # whether the energy weights below are None.
    if force_aggregation not in ("structure_mean", "atom_component_pooled"):
        raise ValueError(
            f"rank_force_fields_combined: force_aggregation={force_aggregation!r} is not implemented -- "
            f"expected 'structure_mean' or 'atom_component_pooled' (see this function's own docstring).")

    # weight=1.0/1.0 here (not the caller's own weights) -- these calls
    # produce RAW, unweighted per-component RMSE; this function's own
    # validation_force_weight/training_force_weight args are applied
    # afterward, once, when combining components -- same "raw first,
    # weight once at the end" pattern as format_ranking_table's own
    # raw_deviation_dct. energy_per_atom is passed through unconditionally
    # (cheap either way -- get_ff_deviations always computes an energy
    # term as a byproduct of computing forces, whether or not this
    # function goes on to use it) so val_deviation_dct/train_deviation_dct
    # carry the right (normalized-or-not) energy value regardless of
    # whether the energy weights below are None. Needed for ENERGY (always)
    # and for FORCE too when force_aggregation=="structure_mean" (the
    # default) -- when force_aggregation=="atom_component_pooled" instead,
    # get_pooled_force_rmse (below) supplies the force values directly and
    # this call's own force values are simply unused for that part.
    val_deviation_dct = get_ff_deviations(validation_ff_dct, validation_reference_dct, 1.0, 1.0,
                                           reference_label=reference_label, energy_per_atom=energy_per_atom)
    train_deviation_dct = get_ff_deviations(training_ff_dct, training_reference_dct, 1.0, 1.0,
                                             reference_label=reference_label, energy_per_atom=energy_per_atom)

    if force_aggregation == "atom_component_pooled":
        pooled_val_force = get_pooled_force_rmse(validation_ff_dct, validation_reference_dct, reference_label)
        pooled_train_force = get_pooled_force_rmse(training_ff_dct, training_reference_dct, reference_label)

    has_validation_energy = validation_energy_weight is not None
    has_training_energy = training_energy_weight is not None
    has_training_rel_e = training_relative_energy_dct is not None
    has_validation_rel_e = validation_relative_energy_dct is not None

    if has_training_rel_e:
        training_rel_e_scores = get_relative_energy_scores(
            get_relative_energy_deviations(training_relative_energy_dct, training_relative_energy_reference_dct, reference_label))
    if has_validation_rel_e:
        validation_rel_e_scores = get_relative_energy_scores(
            get_relative_energy_deviations(validation_relative_energy_dct, validation_relative_energy_reference_dct, reference_label))

    components = {}
    for label in validation_ff_dct:
        val_images = [i for md in val_deviation_dct[label].values() for i in md.values()]
        train_images = [i for md in train_deviation_dct[label].values() for i in md.values()]

        if force_aggregation == "atom_component_pooled":
            mean_val_force = pooled_val_force[label]
            mean_train_force = pooled_train_force[label]
        else:
            mean_val_force = sum((i['fx'] + i['fy'] + i['fz']) / 3 for i in val_images) / len(val_images)
            mean_train_force = sum((i['fx'] + i['fy'] + i['fz']) / 3 for i in train_images) / len(train_images)

        combined = (validation_force_weight * mean_val_force
                    + training_force_weight * mean_train_force)

        components[label] = {
            'mean_val_force': mean_val_force,
            'mean_train_force': mean_train_force,
            'combined': combined,
        }

        if has_validation_energy:
            mean_val_energy = sum(i['energy'] for i in val_images) / len(val_images)
            components[label]['mean_val_energy'] = mean_val_energy
            components[label]['combined'] += validation_energy_weight * mean_val_energy
        if has_training_energy:
            mean_train_energy = sum(i['energy'] for i in train_images) / len(train_images)
            components[label]['mean_train_energy'] = mean_train_energy
            components[label]['combined'] += training_energy_weight * mean_train_energy

        if has_training_rel_e:
            components[label]['mean_train_relative_energy'] = training_rel_e_scores[label]
            components[label]['combined'] += training_relative_energy_weight * training_rel_e_scores[label]
        if has_validation_rel_e:
            components[label]['mean_val_relative_energy'] = validation_rel_e_scores[label]
            components[label]['combined'] += validation_relative_energy_weight * validation_rel_e_scores[label]

    sorted_labels = sorted(components, key=lambda l: components[l]['combined'])

    # ff_label column is sized to the longest actual label (not a fixed
    # guess) -- real labels like "reaxff_run_14_bonds_and_offdiag" are far
    # wider than a fixed width, which was misaligning every row.
    label_width = max([len('ff_label')] + [len(l) for l in sorted_labels])

    # Extra columns are appended, in this fixed order, only for whichever
    # relative-energy dcts were actually passed -- keeps the table/header
    # in sync without hardcoding which columns exist.
    extra_columns = []
    if has_validation_energy:
        extra_columns.append(('mean_val_energy', 'val_energy_dev', 24))
    if has_training_energy:
        extra_columns.append(('mean_train_energy', 'train_energy_dev', 24))
    if has_training_rel_e:
        extra_columns.append(('mean_train_relative_energy', 'train_rel_e_dev_kcalmol', 24))
    if has_validation_rel_e:
        extra_columns.append(('mean_val_relative_energy', 'val_rel_e_dev_kcalmol', 24))

    header = (f"{'rank':>4}  {'ff_label':<{label_width}}  {'val_force_dev_kcalmolA':>23}  "
              f"{'train_force_dev_kcalmolA':>25}")
    for _, column_label, width in extra_columns:
        header += f"  {column_label:>{width}}"
    header += f"  {'combined_score':>15}"

    lines = [header]
    for rank, label in enumerate(sorted_labels, start=1):
        c = components[label]
        line = (f"{rank:>4}  {label:<{label_width}}  {c['mean_val_force']:>23.6f}  "
                f"{c['mean_train_force']:>25.6f}")
        for key, _, width in extra_columns:
            line += f"  {c[key]:>{width}.6f}"
        line += f"  {c['combined']:>15.6f}"
        lines.append(line)

    return lines, sorted_labels, [components[l]['combined'] for l in sorted_labels]

def assert_comparable_dicts(
    ff_dct,
    reference_dct,
    reference_label,
):
    """
    Ensure ff_dct and reference_dct[reference_label] share the same
    md_name / md_image structure.
    """
    if reference_label not in reference_dct:
        raise KeyError(f"Reference label '{reference_label}' not found in reference_dct")

    ref_root = reference_dct[reference_label]

    for ff_label, ff_label_dct in ff_dct.items():
        for md_name, md_dct in ff_label_dct.items():
            if md_name not in ref_root:
                raise KeyError(
                    f"[{ff_label}] md_name '{md_name}' missing in reference dictionary"
                )

            for md_image in md_dct:
                if md_image not in ref_root[md_name]:
                    raise KeyError(
                        f"[{ff_label}] md_image '{md_image}' missing in reference "
                        f"for md_name '{md_name}'"
                    )

def get_pooled_force_rmse(ff_dct, reference_dct, reference_label="DFT"):
    """
    ADDED -- a second, genuinely different force-deviation convention from
    get_ff_deviations' own ("structure_mean": one RMSE per structure,
    averaged uniformly across structures). This one pools every individual
    force COMPONENT (fx, fy, fz) from every atom of every structure for a
    given label into ONE flat array, then takes a single RMSE over that --
    implicitly atom-count-weighted (a 200-atom structure contributes 600
    values to the pool, a 20-atom one only 60), NOT structure-count-weighted.

    Exists because these two conventions are NOT interchangeable -- a real,
    confirmed ~25-35% discrepancy between them was found and documented by
    a downstream caller (the companion Pd-Se ACE-fitting pipeline's own
    rank_ace_validation) comparing against pyace/tensorpotential's own
    fitting-time test_metrics.txt, which uses THIS pooling convention, not
    get_ff_deviations' own. Reproduces tensorpotential.fitmetrics.py's own
    compute_metrics pooling exactly (read directly from this project's own
    TensorPotential fork, not reimplemented from a guess) -- the companion
    pipeline's own version of this exact logic was independently verified
    to reproduce real test_metrics.txt rmse_f_comp to 8 significant figures.

    Added as a SEPARATE function rather than a mode on get_ff_deviations
    itself: that function's existing per-(md_name, md_image) return shape
    is relied on by other callers (e.g. write_per_structure_force_
    deviations) that need genuinely PER-STRUCTURE values -- pooling
    everything into one number per label doesn't fit that shape at all, so
    reusing it as a parameterized mode would mean two incompatible return
    shapes behind one signature. This function has a deliberately narrower,
    single-purpose contract instead: returns {label: rmse_force_pooled}.

    Expected to grow more pooling/aggregation conventions over time (this
    project's own anticipated use: extending this ranking to other MLIP/
    interatomic-potential backends, which may report deviations via yet
    other conventions) -- see rank_force_fields_combined's own
    force_aggregation parameter, which dispatches to whichever convention
    is requested rather than hardcoding one.
    """
    assert_comparable_dicts(ff_dct, reference_dct, reference_label)
    ref_root = reference_dct[reference_label]

    out = {}
    for label, md_dct in ff_dct.items():
        df_components = []
        for md_name, image_dct in md_dct.items():
            for image, props in image_dct.items():
                ref_props = ref_root[md_name][image]
                for axis in ('fx', 'fy', 'fz'):
                    df_components.append(np.asarray(ref_props[axis]) - np.asarray(props[axis]))
        df_components = np.concatenate(df_components)
        out[label] = float(np.sqrt(np.mean(df_components ** 2)))
    return out


def get_ff_deviations(
    ff_dct,
    reference_dct,
    energy_weight,
    force_weight,
    reference_label="DFT",
    energy_per_atom=False,
):
    """
    Compute weighted energy + force deviations of each FF
    relative to a reference.

    energy_per_atom (ADDED, default False -- preserves every existing
    caller's behavior unchanged): if True, the energy term is normalized
    by atom count (len(ref_dct["fx"]), the same per-structure atom count
    the force arrays already imply -- no separate atom-count field needed)
    before computing RMSE, i.e. RMSE(ref_energy/natoms, ff_energy/natoms)
    instead of RMSE(ref_energy, ff_energy). Needed by callers (e.g. ACE
    validation) whose own energy convention is already per-atom
    (energy_corrected/natoms) throughout, unlike ReaxFF's own raw total-
    energy convention, which this function's default (False) still matches
    exactly.
    """
    assert_comparable_dicts(ff_dct, reference_dct, reference_label)

    ff_deviation_dct = {}

    ref_root = reference_dct[reference_label]

    for ff_label, ff_label_dct in ff_dct.items():
        ff_deviation_dct[ff_label] = {}

        for md_name, md_dct in ff_label_dct.items():
            ff_deviation_dct[ff_label][md_name] = {}

            for md_image, i_dct in md_dct.items():
                ref_dct = ref_root[md_name][md_image]

                # Scalars
                ref_energy, ff_energy = ref_dct["energy"], i_dct["energy"]
                if energy_per_atom:
                    natoms = len(ref_dct["fx"])
                    ref_energy, ff_energy = ref_energy / natoms, ff_energy / natoms
                w_e = energy_weight * root_mean_squared_error(
                    [ref_energy],
                    [ff_energy],
                )

                # Forces (arrays)
                fx_ref, fy_ref, fz_ref = map(np.asarray, (ref_dct["fx"], ref_dct["fy"], ref_dct["fz"]))
                fx, fy, fz = map(np.asarray, (i_dct["fx"], i_dct["fy"], i_dct["fz"]))

                w_fx = force_weight * root_mean_squared_error(fx_ref, fx)
                w_fy = force_weight * root_mean_squared_error(fy_ref, fy)
                w_fz = force_weight * root_mean_squared_error(fz_ref, fz)

                summed = w_e + w_fx + w_fy + w_fz

                ff_deviation_dct[ff_label][md_name][md_image] = {
                    "summed": summed,
                    "energy": w_e,
                    "fx": w_fx,
                    "fy": w_fy,
                    "fz": w_fz,
                }

    return ff_deviation_dct


def write_per_structure_force_deviations(
    ff_dct, reference_dct, output_root,
    reference_label="DFT",
    filename="force_deviations.csv",
):
    """
    Writes one file per force field, into output_root/<ff_label>/filename
    (that force field's own run directory, alongside its ffield/
    structures/ -- e.g. MD/single_points/reaxff_validation/aimd/
    force_fields/reaxff_run_0_angular_only/force_deviations.csv), with one
    row per structure (md_name/md_image): its per-structure, UNWEIGHTED
    fx/fy/fz force RMSE, in kcal/mol/Angstrom (see get_ff_deviations,
    called here with energy_weight=force_weight=1.0 for the same reason
    rank_force_fields_combined's own raw_deviation_dct does -- a
    configured weight of 0 would make recovering the raw value from a
    weighted one undefined; ff_dct/reference_dct are assumed already
    unit-converted to kcal/mol/kcal-mol-per-Angstrom throughout, same
    assumption as rank_force_fields_combined -- see
    convert_units_to_kcal_mol).

    No energy column -- this used to optionally include one (via a
    frame-0-relative "energy" deviation, for validation's AIMD
    trajectories specifically), but that was a plain-subtraction
    comparison with no stoichiometric (get_divisors) reduction, a
    different and less rigorous method than the YAML-driven relative-
    energy comparison (relative_energy_comparison.py's own
    relative_energy_comparison.csv) now covers for both training AND
    validation consistently -- see rank_force_fields_combined's own
    docstring for the same reasoning. Energy accuracy belongs in that
    file; this one is forces only.

    Recomputes get_ff_deviations independently from rank_force_fields_
    combined's own internal call (a little duplicated work, not reused
    via a shared return value) -- keeps rank_force_fields_combined's
    existing return signature/contract untouched rather than threading a
    detail dict through it for what's otherwise a self-contained,
    additive output.
    """
    deviation_dct = get_ff_deviations(ff_dct, reference_dct, 1.0, 1.0, reference_label=reference_label)

    header = "structure,fx_dev_kcalmolA,fy_dev_kcalmolA,fz_dev_kcalmolA"

    for ff_label, md_dct in tqdm(deviation_dct.items(), desc="writing force deviations", unit="force_field"):
        lines = [header]
        for md_name, image_dct in md_dct.items():
            for md_image, dev in image_dct.items():
                structure = f"{md_name}/{md_image}" if md_name else md_image
                lines.append(f"{structure},{dev['fx']:.6f},{dev['fy']:.6f},{dev['fz']:.6f}")

        out_path = os.path.join(output_root, ff_label, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

def get_ff_scores(deviation_dct):
    """
    Average deviation score per force field.
    """
    score_dct = {}

    for ff_label, ff_label_dct in deviation_dct.items():
        total = 0.0
        count = 0

        for md_dct in ff_label_dct.values():
            for image_dct in md_dct.values():
                total += image_dct["summed"]
                count += 1

        if count == 0:
            raise ValueError(f"No images found for force field '{ff_label}'")

        score_dct[ff_label] = total / count

    return score_dct

def rank_ff_scores(
    ff_dct,
    reference_dct,
    energy_weight,
    force_weight,
    reference_label="DFT",
    reverse=False
):
    """
    Rank force fields by average deviation score.
    Lower is better by default.
    """

    deviation_dct = get_ff_deviations(ff_dct,
                                      reference_dct,
                                      energy_weight,
                                      force_weight,
                                      reference_label=reference_label)

    score_dct = get_ff_scores(deviation_dct)

    sorted_items = sorted(
        score_dct.items(),
        key=lambda x: x[1],
        reverse=reverse,
    )

    ff_labels, scores = map(list, zip(*sorted_items))
    return ff_labels, scores

def format_ranking_table(ff_dct, reference_dct, energy_weight, force_weight, reference_label="DFT"):
    """
    Build, per force field ranked best-to-worst (lower weighted score is
    better), a table of lines: mean raw (unweighted) energy RMSE, mean raw
    force RMSE (average of fx/fy/fz), and the weighted summed score
    actually used for ranking/selection. Raw deviations come from a
    separate energy_weight=1/force_weight=1 call rather than dividing the
    weighted output back out, since a configured weight of 0 would make
    that division undefined. Returns (lines, labels, scores), the latter
    two in ranked order.
    """
    raw_deviation_dct = get_ff_deviations(ff_dct, reference_dct, 1.0, 1.0, reference_label=reference_label)
    labels, scores = rank_ff_scores(ff_dct, reference_dct, energy_weight, force_weight, reference_label=reference_label)

    lines = [f"{'rank':>4}  {'ff_label':>10}  {'raw_energy_rmse':>16}  {'raw_force_rmse':>16}  {'weighted_score':>15}"]
    for rank, (label, score) in enumerate(zip(labels, scores), start=1):
        raw = raw_deviation_dct[label]
        n = sum(len(images) for images in raw.values())
        raw_e = sum(i["energy"] for md in raw.values() for i in md.values()) / n
        raw_f = sum((i["fx"] + i["fy"] + i["fz"]) / 3 for md in raw.values() for i in md.values()) / n
        lines.append(f"{rank:>4}  {label:>10}  {raw_e:>16.6f}  {raw_f:>16.6f}  {score:>15.6f}")

    return lines, labels, scores
