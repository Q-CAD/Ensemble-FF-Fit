#!/usr/bin/env python3

from multiprocessing import Pool, cpu_count
from functools import partial
import os
import argparse
import numpy as np
import sys
from pymatgen.core import Structure, Lattice
from EnsembleFFFit.analysis.properties import parse_single_points
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Generate new POSCAR files from MD with Uncertainty Quantification"
    )
    parser.add_argument("--md_directory", "-md", required=True)
    parser.add_argument("--max_structures", "-max", type=int, default=10)
    parser.add_argument("--cutout", action="store_true")
    parser.add_argument("--cutout_dimensions", type=float, default=10)
    parser.add_argument("--low_high", action="store_true")
    parser.add_argument("--force_weight", "-fw", type=float, default=1.0)
    parser.add_argument("--energy_weight", "-ew", type=float, default=1.0)
    parser.add_argument("--write_directory", "-wd", required=True)

    args = parser.parse_args()
    write_new_structures(args)

def _cutout_worker(args):
    structure, site_index, score, box_length = args
    cutout = cubic_realspace_cutout_bruteforce(
        structure, site_index, box_length, image_range=1
    )
    return cutout, score


def site_structure_score(structure_dct, site_variance_dct, box_length):
    tasks = []

    for key, s_dct in structure_dct.items():
        structure = s_dct["structure"]
        site_sums = site_variance_dct[key]["site_sums"]

        for i, score in enumerate(site_sums):
            tasks.append((structure, i, score, box_length))

    structures, scores = [], []

    with Pool(processes=cpu_count()) as pool:
        for cutout, score in tqdm(
            pool.imap_unordered(_cutout_worker, tasks),
            total=len(tasks),
            desc="Building cutouts",
        ):
            structures.append(cutout)
            scores.append(score)

    return structures, scores


def base_structure_score(structure_dct, site_variance_dct):
    structures, scores = [], []

    for key, s_dct in tqdm(structure_dct.items(), desc="Processing structures"):
        structures.append(s_dct["structure"])
        scores.append(site_variance_dct[key]["average"])

    return structures, scores


def get_structures_scores(dct, args):
    aaa = AseAtomsAdaptor()
    structure_dct = {}

    for ff_dct in dct.values():
        for s_dct in ff_dct.values():
            for image_key, i_dct in s_dct.items():
                structure = aaa.get_structure(i_dct["atoms"])

                entry = structure_dct.setdefault(
                    image_key,
                    {
                        "energies": [],
                        "atomic_energies": [],
                        "fxs": [],
                        "fys": [],
                        "fzs": [],
                        "structure": structure,
                    },
                )

                entry["energies"].append(i_dct["energy"])
                entry["atomic_energies"].append(i_dct["e_atoms"])
                entry["fxs"].append(i_dct["fx"])
                entry["fys"].append(i_dct["fy"])
                entry["fzs"].append(i_dct["fz"])

    site_variance_dct = {}

    for key, s_dct in structure_dct.items():
        w_e = args.energy_weight * np.var(s_dct["atomic_energies"], axis=0)
        w_fx = args.force_weight * np.var(s_dct["fxs"], axis=0)
        w_fy = args.force_weight * np.var(s_dct["fys"], axis=0)
        w_fz = args.force_weight * np.var(s_dct["fzs"], axis=0)

        site_sums = w_e + w_fx + w_fy + w_fz

        site_variance_dct[key] = {
            "site_sums": site_sums,
            "average": float(np.mean(site_sums)),
        }

    if args.cutout:
        return site_structure_score(structure_dct, site_variance_dct, args.cutout_dimensions)
    else:
        return base_structure_score(structure_dct, site_variance_dct)


def rank_structures(args):
    all_structures, all_scores = [], []
    index = -1

    while True:
        try:
            dct = parse_single_points(args.md_directory, index, ffield_label=(-4, -2))
            structures, scores = get_structures_scores(dct, args)
            all_structures.extend(structures)
            all_scores.extend(scores)
            index -= 1
        except IndexError:
            break

    reverse = False if args.low_high is True else True
    sorted_pairs = sorted(
        zip(all_structures, all_scores), key=lambda x: x[1], reverse=reverse
    )

    return [s for s, _ in sorted_pairs[: args.max_structures]]


def cubic_realspace_cutout_bruteforce(
    structure, site_index, box_length, image_range=1
):
    center = structure[site_index].coords
    half = box_length / 2.0

    bounds_min = center - half
    bounds_max = center + half

    species, coords = [], []
    lat = structure.lattice

    for i in range(-image_range, image_range + 1):
        for j in range(-image_range, image_range + 1):
            for k in range(-image_range, image_range + 1):
                shift = np.array([i, j, k])
                for site in structure:
                    f = site.frac_coords + shift
                    c = lat.get_cartesian_coords(f)

                    if np.all(c >= bounds_min) and np.all(c <= bounds_max):
                        species.append(site.specie)
                        coords.append(c - center + half)

    return Structure(
        lattice=Lattice.cubic(box_length),
        species=species,
        coords=coords,
        coords_are_cartesian=True,
    )


def write_new_structures(args):
    top_structures = rank_structures(args)

    for i, structure in enumerate(top_structures):
        write_path = os.path.join(args.write_directory, str(i))
        os.makedirs(write_path, exist_ok=True)
        structure.sort().to(
            fmt="poscar",
            filename=os.path.join(write_path, "POSCAR"),
        )


if __name__ == "__main__":
    main()

