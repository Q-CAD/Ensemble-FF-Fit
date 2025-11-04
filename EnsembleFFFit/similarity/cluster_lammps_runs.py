#!/usr/bin/env python3
"""
cluster_by_similarity.py

Dependencies:
  - numpy
  - scipy
  - matminer
  - ase
  - pymatgen
  - EnsembleFFFit (for parse_single_points)
"""

from pathlib import Path
import sys
import re
import json
import os
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

from EnsembleFFFit.analysis.properties import parse_single_points
#from parallel_feat import parallel_featurize_structures_local

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import numpy as np
import multiprocessing

# NOTE: matminer imports are inside initializer to avoid pickling
_global_ssf = None

def _init_worker_ssf():
    """
    Worker initializer: build the SiteStatsFingerprint in each child process.
    This avoids attempting to pickle the featurizer into the pool.
    """
    global _global_ssf
    from matminer.featurizers.site import CrystalNNFingerprint
    from matminer.featurizers.structure import SiteStatsFingerprint
    # create the featurizer (OPS preset)
    cnn = CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0)
    _global_ssf = SiteStatsFingerprint(cnn, stats=("mean", "std_dev", "minimum", "maximum"))

def _worker_featurize(index_and_structure):
    """
    Worker function: featurize a single structure using the per-process _global_ssf.
    Receives a tuple (index, structure) so ordering can be recovered (though map keeps order).
    Returns (index, feature_array).
    """
    idx, struct = index_and_structure
    global _global_ssf
    if _global_ssf is None:
        # Fallback - shouldn't happen if initializer used
        _init_worker_ssf()
    try:
        feat = _global_ssf.featurize(struct)
    except Exception:
        # fallback if API differs
        feat = _global_ssf.feature(struct)
    return idx, np.asarray(feat, dtype=float)

def detect_num_workers(max_workers=None, prefer_env=True):
    """
    Decide how many workers to spawn.
    If running under SLURM and prefer_env True, honor SLURM_CPUS_ON_NODE or SLURM_JOB_CPUS_PER_NODE.
    Otherwise use os.cpu_count() (or supplied max_workers).
    """
    if max_workers is not None:
        return int(max_workers)

    # prefer SLURM env vars if present
    if prefer_env:
        for var in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_NODE", "SLURM_JOB_CPUS_PER_NODE"):
            val = os.environ.get(var)
            if val:
                try:
                    return max(1, int(val))
                except ValueError:
                    pass

    # fallback to available CPUs
    n = os.cpu_count() or 1
    return max(1, int(n))

def parallel_featurize_structures_local(structures,
                                        num_workers=None,
                                        verbose=True):
    """
    Featurize a list of pymatgen.Structure objects (or compatible) in parallel
    using local multiprocessing. Returns a stacked numpy array (N, F) in the same order.

    - structures: list-like
    - num_workers: override automatic worker count
    - verbose: show tqdm progress
    """
    N = len(structures)
    if N == 0:
        return np.zeros((0, 0))

    n_workers = detect_num_workers(num_workers)
    # don't spawn more workers than structures
    if n_workers > N:
        n_workers = N

    # Create index+structure iterable so we can be extra safe about ordering
    indexed = ((i, structures[i]) for i in range(N))

    # Use ProcessPoolExecutor with initializer to build featurizer in each process
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker_ssf) as ex:
        # executor.map preserves the order of the input iterable
        # but we want a progress bar: wrap the map iterator with tqdm
        # map returns results in order so the stacking will be aligned.
        it = ex.map(_worker_featurize, indexed)
        if verbose:
            it = tqdm(it, total=N, desc=f"Featurizing ({n_workers} workers)")
        results = list(it)  # list of (idx, feat)

    # results are in input order because executor.map preserves order.
    # But for double safety, sort by idx and stack
    results.sort(key=lambda x: x[0])
    feats = [feat for idx, feat in results]
    feats = np.vstack(feats) if feats else np.zeros((0, 0))
    return feats

# -------------------------
# Reference / formation energy helpers
# -------------------------
def get_reference_per_atom(dct):
    """Compute per-atom reference energies (min across configurations)."""
    ref_per_atom = {}
    for ffield, sub in dct.items():
        for element, entries in sub.items():
            energies = []
            for subkey, props in entries.items():
                atoms = props["atoms"]
                e_tot = props["energy"]
                n_atoms = len(atoms)
                energies.append(e_tot / n_atoms)
            ref_per_atom[element] = np.min(energies)
    print("Reference per-atom energies:", ref_per_atom)
    return ref_per_atom


def get_formation_energy_data(dct, ref_dct):
    """
    Build a sorted list of formation energies.

    Returns list of tuples:
      (label, real_formula, subkey, atoms, formation_energy_per_atom)
    """
    formation_data = []
    for ffield, sub in dct.items():
        for formula, entries in sub.items():
            for subkey, props in entries.items():
                atoms = props["atoms"]
                e_tot = props["energy"]

                # Get composition (dict of element -> count)
                symbols, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
                s_symbols = [str(s) for s in symbols]
                composition = dict(zip(s_symbols, counts))
                real_formula = atoms.get_chemical_formula()

                # Total number of atoms in the system
                n_total = len(atoms)

                # Reference energy for that composition
                e_ref = 0.0
                for elem, count in composition.items():
                    if elem not in ref_dct:
                        raise KeyError(f"Missing reference for element {elem}")
                    e_ref += ref_dct[elem] * count

                # Formation energy per atom
                try:
                    e_form = (e_tot - e_ref) / n_total
                    formation_data.append((formula, real_formula, subkey, atoms, e_tot, e_form))
                except TypeError:
                    print(f"Energy of {e_tot} for {formula}")

    # Sort ascending by formation energy
    formation_data.sort(key=lambda x: x[-1])

    return formation_data


# -------------------------
# SFPD dissimilarity helper
# -------------------------
def build_sfpd_featurizer():
    """
    Return a SiteStatsFingerprint object configured for SFPD (OPS preset).
    Reuse the same featurizer for speed.
    """
    cnn = CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0)
    ssf = SiteStatsFingerprint(cnn, stats=("mean", "std_dev", "minimum", "maximum"))
    return ssf


def compute_pairwise_dissimilarity_pmg(structures, ssf=None, verbose=False, parallel=True):
    """
    Compute NxN symmetric dissimilarity matrix for a list of pymatgen.Structure
    objects using SFPD (site-fingerprint pairwise distance).
    Returns a numpy array shape (N,N).
    """
    n = len(structures)
    if n == 0: 
        return np.zeros((0, 0))

    if parallel:
        feats = parallel_featurize_structures_local(structures, num_workers=None, verbose=True)
    else:
        if ssf is None:
            ssf = build_sfpd_featurizer()

        # Precompute features for each structure
        feats = []
        for i, s in tqdm(enumerate(structures), desc="Computing SFPD"):
            try:
                feat = ssf.featurize(s)
            except Exception:
                try:
                    feat = ssf.feature(s)
                except Exception as exc:
                    raise RuntimeError(f"Failed to featurize structure index {i}: {exc}")
            feats.append(np.asarray(feat, dtype=float))

    feats = np.vstack(feats)  # shape (N, F)

    # Pairwise Euclidean distances between feature vectors (vectorized)
    norms = np.sum(feats * feats, axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * feats.dot(feats.T)
    d2 = np.clip(d2, 0.0, None)
    D = np.sqrt(d2)
    np.fill_diagonal(D, 0.0)

    if verbose:
        print(f"Computed pairwise dissimilarity matrix for {n} structures.")
    return D


# -------------------------
# clustering + representative selection
# -------------------------
def cluster_and_choose_representatives(formation_list,
                                       energy_cutoff=1.0,
                                       distance_threshold=0.5,
                                       max_structures=None,
                                       outdir="cluster_output",
                                       verbose=False):
    """
    Cluster formation_list and choose representative structures per cluster.

    Parameters:
      formation_list: list of (label, real_formula, subkey, atoms, e_form)
      energy_cutoff: include only structures with e_form <= energy_cutoff (eV/atom)
      distance_threshold: clustering threshold (distance) for fcluster
      max_structures: cap the number of structures passed to clustering (lowest energy)
      outdir: directory to write POSCARs, a folder per cluster
      verbose: print progress

    Returns:
      cluster_results: dict mapping cluster_id -> info dict
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Filter by energy cutoff
    filtered = [(i, *item) for i, item in enumerate(formation_list) if item[-1] <= energy_cutoff]
    if not filtered:
        raise ValueError("No structures below the energy cutoff.")

    filtered.sort(key=lambda x: x[-1])  # ascending by formation energy

    # 2) Optionally reduce number of structures
    if max_structures is not None and len(filtered) > max_structures:
        if verbose:
            print(f"Truncating {len(filtered)} -> {max_structures} lowest-energy structures for clustering.")
        filtered = filtered[:max_structures]

    # Extract lists for clustering + keep mapping back to original data
    indices = [item[0] for item in filtered]   # original index in formation_list
    labels = [item[1] for item in filtered]
    formulas = [item[2] for item in filtered]
    subkeys = [item[3] for item in filtered]
    atoms_list = [item[4] for item in filtered]
    abs_energies = [item[5] for item in filtered]
    energies = [item[6] for item in filtered]

    n = len(atoms_list)
    if verbose:
        print(f"Clustering {n} structures (energy cutoff = {energy_cutoff} eV/atom)")

    # 3) Convert ASE Atoms -> pymatgen Structure for matminer
    aaa = AseAtomsAdaptor()
    pmg_structures = [aaa.get_structure(a) for a in atoms_list]

    # 4) Compute pairwise dissimilarity matrix (SFPD)
    ssf = build_sfpd_featurizer()
    D = compute_pairwise_dissimilarity_pmg(pmg_structures, ssf=ssf, verbose=verbose)

    # 5) Hierarchical clustering
    if n == 1:
        cluster_ids = np.array([1], dtype=int)
    else:
        condensed = squareform(D)
        Z = linkage(condensed, method="single")
        cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")

    # Build clusters mapping
    clusters = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)

    # 6) Choose representative per cluster (lowest formation energy)
    cluster_results = {}
    for cid, member_inds in clusters.items():
        member_energies = [energies[i] for i in member_inds]
        best_local_idx = member_inds[int(np.argmin(member_energies))]

        rep_info = {"cluster_id": int(cid), "members": [], "representative": None}
        for i in member_inds:
            rep_entry = {
                "global_index": int(indices[i]),
                "label": labels[i],
                "formula": formulas[i],
                "subkey": str(subkeys[i]),
                "abs_energy": float(abs_energies[i]),
                "energy": float(energies[i]),
            }
            rep_info["members"].append(rep_entry)

        rep_entry = {
            "global_index": int(indices[best_local_idx]),
            "label": labels[best_local_idx],
            "formula": formulas[best_local_idx],
            "subkey": str(subkeys[best_local_idx]),
            "abs_energy": float(abs_energies[i]),
            "energy": float(energies[best_local_idx]),
            "atoms": atoms_list[best_local_idx],
        }
        rep_info["representative"] = rep_entry
        cluster_results[int(cid)] = rep_info

    # helper to sanitize filenames
    def _sanitize_filename(s: str) -> str:
        s = str(s)
        s = re.sub(r"[\/\s]+", "_", s)
        s = re.sub(r"[^0-9A-Za-z_\-\.]", "", s)
        return s[:200]

    # 7) Write POSCARs grouped by reduced formula subfolders
    for cid, info in cluster_results.items():
        cluster_dir = outdir / f"cluster_{cid:03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Group members by reduced chemical formula
        formula_groups = defaultdict(list)
        for m in info["members"]:
            try:
                local_idx = indices.index(m["global_index"])
            except ValueError:
                continue
            atoms_obj = atoms_list[local_idx]
            pmg_obj = aaa.get_structure(atoms_obj)
            reduced_formula = pmg_obj.composition.reduced_formula
            entry = {
                "member": m,
                "atoms": atoms_obj,
                "energy": float(m["energy"]),
                "abs_energy": float(m["abs_energy"]),
                "local_idx": local_idx,
            }
            formula_groups[reduced_formula].append(entry)

        # assign letter labels (a, b, c...) sorted by group size
        unique_formulas = sorted(formula_groups.keys(), key=lambda f: (-len(formula_groups[f]), f))
        letters = [chr(ord("a") + i) for i in range(len(unique_formulas))]
        formula_to_label = {fmt: letters[i] for i, fmt in enumerate(unique_formulas)}

        # For each formula subgroup, write members & representative JSON
        for fmt, entries in formula_groups.items():
            sublabel = formula_to_label[fmt]
            #subdir_name = f"{sublabel}_{_sanitize_filename(fmt)}"
            subdir_name = f"{_sanitize_filename(fmt)}"
            subdir = cluster_dir / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)

            energy_dict = {}

            # write all members
            for ent in entries:
                m = ent["member"]
                atoms_obj = ent["atoms"]
                pmg_obj = aaa.get_structure(atoms_obj).sort()
                fname = f"{m['label']}_{m['formula']}_{m['subkey']}"
                safe_fname = _sanitize_filename(fname)
                outpath = subdir / safe_fname
                outpath.mkdir(parents=True, exist_ok=True)
                #write(str(os.path.join(outpath, 'POSCAR.vasp')), images=[atoms_obj], format="vasp")
                pmg_obj.to(filename=os.path.join(outpath, 'POSCAR.vasp'), fmt='poscar')
                energy_dict[safe_fname] = {"formation_energy": float(m["energy"]), 
                                           "abs_energy": float(m["abs_energy"])}

            # representative for this formula subgroup (lowest formation energy)
            best_ent = min(entries, key=lambda e: e["energy"])
            rep = best_ent["member"]
            rep_atoms = best_ent["atoms"]
            rep_pmg_obj = aaa.get_structure(rep_atoms).sort()
            rep_fname = f"rep_{rep['label']}_{rep['formula']}_{rep['subkey']}"
            rep_safe = _sanitize_filename(rep_fname)
            rep_path = subdir / rep_safe
            rep_path.mkdir(parents=True, exist_ok=True)
            #write(str(os.path.join(rep_path, 'POSCAR')), images=[rep_atoms], format="vasp")
            rep_pmg_obj.to(filename=os.path.join(rep_path, 'POSCAR'), fmt='poscar')
            energy_dict[rep_safe] = {"formation_energy": float(rep["energy"]), 
                                     "abs_energy": float(rep["abs_energy"])}

            # write energies JSON
            json_path = subdir / "formation_energies.json"
            with open(json_path, "w") as f:
                json.dump(energy_dict, f, indent=2)

        # cluster-level summary
        summary = {
            "cluster_id": int(cid),
            "num_formula_groups": len(formula_groups),
            "formula_to_label": {fmt: formula_to_label[fmt] for fmt in unique_formulas},
        }
        with open(cluster_dir / "cluster_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return cluster_results


# -------------------------
# Example usage (call from your script)
# -------------------------
# formation_sorted is the list you already build:
# formation_sorted = [(label, real_formula, subkey, atoms, e_form), ...]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: cluster_by_similarity.py <references_path> <phases_path>")
        sys.exit(1)

    references_path = sys.argv[1]
    phases_path = sys.argv[2]

    reference_dct = parse_single_points(references_path, dump_index=-1, ffield_label=(-5, -3))
    structure_dct = parse_single_points(phases_path, dump_index=-1, ffield_label=(-5, -3))

    reference_per_atom = get_reference_per_atom(reference_dct)
    formation_sorted = get_formation_energy_data(structure_dct, reference_per_atom)

    for label, formula, subkey, atoms, e_abs, e_form in formation_sorted:
        print(f"{label:15s} {formula:15s} {str(subkey):15s} Formation energy = {e_form:.4f} eV/atom")

    energy_cutoff = 3.0
    result = cluster_and_choose_representatives(
        formation_sorted,
        energy_cutoff=energy_cutoff,
        distance_threshold=0.9,
        max_structures=None,
        outdir=f"output_cluster_{str(energy_cutoff)}",
        verbose=True,
    )

    print(f"Wrote cluster output to: {Path(f'output_cluster_{str(energy_cutoff)}').absolute()}")

