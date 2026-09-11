from sklearn.metrics import root_mean_squared_error
import numpy as np


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


def relativize_energies(run_image_dct, reference_image="0"):
    """
    Returns a NEW {run: {image: props}} dict where every image's "energy"
    is replaced by (that image's own energy - the SAME run's
    reference_image energy) -- i.e. energy relative to each run's own
    starting/reference frame, not an absolute value.

    Needed because ReaxFF is trained on relative energies only, so
    absolute DFT-vs-ReaxFF energy comparison isn't meaningful -- the two
    can differ drastically in absolute terms while their relative
    (frame-to-frame) differences agree well. Comparing each side's own
    relative-to-frame-0 energy series sidesteps this entirely.

    Does not mutate the input; forces are left untouched. Raises KeyError
    if any run is missing its own reference_image.
    """
    out = {}
    for run, images in run_image_dct.items():
        if reference_image not in images:
            raise KeyError(f"run {run!r} has no reference image {reference_image!r} to relativize against")
        ref_energy = images[reference_image]['energy']
        out[run] = {}
        for image, props in images.items():
            converted = dict(props)
            converted['energy'] = props['energy'] - ref_energy
            out[run][image] = converted
    return out


def rank_force_fields_combined(
    validation_ff_dct, validation_reference_dct,
    training_ff_dct, training_reference_dct,
    validation_energy_weight=1.0, validation_force_weight=1.0, training_force_weight=1.0,
    reference_label="DFT", reference_image="0",
):
    """
    Combined ReaxFF validation ranking (see run_pipeline.py's
    run_rank_reaxff_validation docstring for the full pipeline this feeds):

    - validation_ff_dct/validation_reference_dct: {label: {run: {image:
      props}}} / {reference_label: {run: {image: props}}}, from DFT/
      validation's AIMD trajectories. Energies are relativized (see
      relativize_energies) against each run's own reference_image before
      scoring -- ReaxFF is trained on relative energies only, so absolute
      DFT-vs-ReaxFF energy comparison isn't meaningful (see
      relativize_energies' own docstring). Forces are compared directly.
    - training_ff_dct/training_reference_dct: same shape, from DFT/
      training -- forces only, no relative-energy concept applied (no
      sensible reference frame exists for arbitrary non-trajectory
      structures) -- the energy component is computed but never used in
      the combined score.

    Both ff_dct arguments are assumed ALREADY unit-converted to match
    their reference (see convert_units) -- this function does no unit
    conversion itself, only relativizing + RMSE scoring.

    Returns (lines, labels, combined_scores): lines is a ranked table
    (best/lowest combined score first) reporting, per force field, the
    three components this session's own investigation confirmed worth
    inspecting separately -- mean validation energy deviation (relative),
    mean validation force deviation, mean training force deviation -- next
    to the combined weighted score actually used for ranking. labels/
    combined_scores are the same ranking, machine-readable.
    """
    val_ff_relative = {label: relativize_energies(run_dct, reference_image)
                       for label, run_dct in validation_ff_dct.items()}
    val_ref_relative = {reference_label: relativize_energies(
        validation_reference_dct[reference_label], reference_image)}

    # weight=1.0/1.0 here (not the caller's own weights) -- these calls
    # produce RAW, unweighted per-component RMSE; this function's own
    # validation_*_weight/training_force_weight args are applied afterward,
    # once, when combining the three components -- same "raw first, weight
    # once at the end" pattern as format_ranking_table's own raw_deviation_dct.
    val_deviation_dct = get_ff_deviations(val_ff_relative, val_ref_relative, 1.0, 1.0, reference_label=reference_label)
    train_deviation_dct = get_ff_deviations(training_ff_dct, training_reference_dct, 1.0, 1.0, reference_label=reference_label)

    components = {}
    for label in validation_ff_dct:
        val_images = [i for md in val_deviation_dct[label].values() for i in md.values()]
        train_images = [i for md in train_deviation_dct[label].values() for i in md.values()]

        mean_val_energy = sum(i['energy'] for i in val_images) / len(val_images)
        mean_val_force = sum((i['fx'] + i['fy'] + i['fz']) / 3 for i in val_images) / len(val_images)
        mean_train_force = sum((i['fx'] + i['fy'] + i['fz']) / 3 for i in train_images) / len(train_images)

        combined = (validation_energy_weight * mean_val_energy
                    + validation_force_weight * mean_val_force
                    + training_force_weight * mean_train_force)

        components[label] = {
            'mean_val_energy': mean_val_energy,
            'mean_val_force': mean_val_force,
            'mean_train_force': mean_train_force,
            'combined': combined,
        }

    sorted_labels = sorted(components, key=lambda l: components[l]['combined'])

    # ff_label column is sized to the longest actual label (not a fixed
    # guess) -- real labels like "reaxff_run_14_bonds_and_offdiag" are far
    # wider than a fixed width, which was misaligning every row.
    label_width = max([len('ff_label')] + [len(l) for l in sorted_labels])

    lines = [f"{'rank':>4}  {'ff_label':<{label_width}}  {'val_energy_dev':>15}  {'val_force_dev':>15}  "
             f"{'train_force_dev':>16}  {'combined_score':>15}"]
    for rank, label in enumerate(sorted_labels, start=1):
        c = components[label]
        lines.append(f"{rank:>4}  {label:<{label_width}}  {c['mean_val_energy']:>15.6f}  {c['mean_val_force']:>15.6f}  "
                      f"{c['mean_train_force']:>16.6f}  {c['combined']:>15.6f}")

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
