import numpy as np

def format_image_dictionary(single_point_dct):
    """
    Reformat:
    single_point_dct[ff_label][md_name][md_image] -> properties
    into:
    new_dct[md_name][md_image] -> lists of energies and forces across FFs
    """
    new_dct = {}

    for _, ff_dct in single_point_dct.items():
        for md_name, md_dct in ff_dct.items():
            md_entry = new_dct.setdefault(md_name, {})

            for md_image, i_dct in md_dct.items():
                image_entry = md_entry.setdefault(
                    md_image,
                    {
                        "energies": [],
                        "fxs": [],
                        "fys": [],
                        "fzs": [],
                        "structure": i_dct["structure"],  # stored once
                    },
                )

                image_entry["energies"].append(i_dct["energy"])
                image_entry["fxs"].append(i_dct["fx"])
                image_entry["fys"].append(i_dct["fy"])
                image_entry["fzs"].append(i_dct["fz"])

    return new_dct

def base_structure_score(image_dct, site_variance_dct):
    """
    Flatten image_dct and site_variance_dct into aligned lists.
    """
    labels = []
    images = []
    structures = []
    scores = []

    for md_name, md_dct in image_dct.items():
        for md_image, i_dct in md_dct.items():
            labels.append(md_name)
            images.append(md_image)
            structures.append(i_dct["structure"])
            scores.append(site_variance_dct[md_name][md_image]["summed"])

    return labels, images, structures, scores

def get_structures_scores(
    single_point_dct,
    energy_weight,
    force_weight,
    reverse=True,
):
    """
    Compute weighted variance scores and return ordered structures.
    """
    site_variance_dct = {}

    image_dct = format_image_dictionary(single_point_dct)
    for md_name, md_dct in image_dct.items():
        site_variance_dct[md_name] = {}

        for md_image, i_dct in md_dct.items():
            energies = np.asarray(i_dct["energies"])
            fxs = np.asarray(i_dct["fxs"])
            fys = np.asarray(i_dct["fys"])
            fzs = np.asarray(i_dct["fzs"])

            w_e = energy_weight * np.var(energies)

            # variance per atom, averaged over atoms
            w_fx = force_weight * np.mean(np.var(fxs, axis=0))
            w_fy = force_weight * np.mean(np.var(fys, axis=0))
            w_fz = force_weight * np.mean(np.var(fzs, axis=0))

            summed = w_e + w_fx + w_fy + w_fz

            site_variance_dct[md_name][md_image] = {
                "summed": summed,
                "energy": w_e,
                "fx": w_fx,
                "fy": w_fy,
                "fz": w_fz,
            }

    labels, images, structures, scores = base_structure_score(
        image_dct, site_variance_dct
    )

    sorted_values = sorted(
        zip(labels, images, structures, scores),
        key=lambda x: x[-1],
        reverse=reverse,
    )

    s_labels, s_images, s_structures, s_scores = map(list, zip(*sorted_values))

    return s_labels, s_images, s_structures, s_scores

def select_structures(labels, images, structures, scores, total=None,
                      score_cap=None, max_per_label=6,
                      image_distance=1000, unique_run=True):
    """
    Downselect (labels, images, structures, scores) -- as produced by
    get_structures_scores, highest-uncertainty first -- to at most
    `max_per_label` images per label (e.g. per MD run), each at least
    `image_distance` apart (as plain integers) from any other selected
    image sharing that label, stopping once `total` images are selected.

    `total=None`/`score_cap=None` mean "no cap" -- pass explicit numbers to
    bound them. This differs from a hardcoded numeric default (e.g.
    score_cap=10) because there's no single score_cap that's sane across
    different energy_weight/force_weight choices or systems, and silently
    dropping the highest-uncertainty images (the very ones this function
    exists to surface) on a stale numeric default would be worse than not
    capping at all.
    """
    if not unique_run:
        n = total if total is not None else len(labels)
        return labels[:n], images[:n], structures[:n], scores[:n]

    selected_labels, selected_images, selected_structures, selected_scores = [], [], [], []

    for i, label in enumerate(labels):
        if total is not None and len(selected_labels) >= total:
            break

        if score_cap is not None and scores[i] > score_cap:
            continue

        if selected_labels.count(label) >= max_per_label:
            continue

        if image_distance > 0:
            too_close = any(
                sel_label == label and abs(int(images[i]) - int(selected_images[j])) < image_distance
                for j, sel_label in enumerate(selected_labels)
            )
            if too_close:
                continue

        selected_labels.append(label)
        selected_images.append(images[i])
        selected_structures.append(structures[i])
        selected_scores.append(scores[i])

    return selected_labels, selected_images, selected_structures, selected_scores

def format_candidate_table(labels, images, scores, output_dirs):
    """Build the (output_dir -> source md_run/frame/score) table as a list
    of lines -- numbered output directories alone (0, 1, 2, ...) don't say
    which trajectory/frame each POSCAR came from, so this is the record of
    that provenance."""
    lines = [f"{'rank':>4}  {'output_dir':>10}  {'md_run':>60}  {'frame':>6}  {'variance_score':>15}"]
    for rank, (label, image, score, out_dir) in enumerate(zip(labels, images, scores, output_dirs), start=1):
        lines.append(f"{rank:>4}  {out_dir:>10}  {label:>60}  {image:>6}  {score:>15.6f}")
    return lines
