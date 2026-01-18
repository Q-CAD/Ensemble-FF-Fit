from sklearn.metrics import root_mean_squared_error
import numpy as np

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
