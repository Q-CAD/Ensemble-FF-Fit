from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
import numpy as np
import os


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
    # raw_deviation_dct.
    val_deviation_dct = get_ff_deviations(validation_ff_dct, validation_reference_dct, 1.0, 1.0, reference_label=reference_label)
    train_deviation_dct = get_ff_deviations(training_ff_dct, training_reference_dct, 1.0, 1.0, reference_label=reference_label)

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

        mean_val_force = sum((i['fx'] + i['fy'] + i['fz']) / 3 for i in val_images) / len(val_images)
        mean_train_force = sum((i['fx'] + i['fy'] + i['fz']) / 3 for i in train_images) / len(train_images)

        combined = (validation_force_weight * mean_val_force
                    + training_force_weight * mean_train_force)

        components[label] = {
            'mean_val_force': mean_val_force,
            'mean_train_force': mean_train_force,
            'combined': combined,
        }

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

def get_ff_deviations(
    ff_dct,
    reference_dct,
    energy_weight,
    force_weight,
    reference_label="DFT",
):
    """
    Compute weighted energy + force deviations of each FF
    relative to a reference.
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
                w_e = energy_weight * root_mean_squared_error(
                    [ref_dct["energy"]],
                    [i_dct["energy"]],
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
