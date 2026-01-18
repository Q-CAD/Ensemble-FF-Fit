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
