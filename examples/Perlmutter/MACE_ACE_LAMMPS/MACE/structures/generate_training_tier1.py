"""
Tier-1 training-structure generator: random crystals across a composition
sweep, via pyxtal. Ported as-is (no behavior changes) from the companion
Pd-Se pipeline's own MACE/finite_temperature/generate_structures/
pyxtal_generate.py.

Standalone, swappable driver -- referenced from workflow_config.yaml's
generate_structures_tier1.generation_task and dynamically imported via
EnsembleFFFit.utilities.general.import_module_from_path, so a future
non-pyxtal generation method is a config change, not a run_pipeline.py
change.

For every (composition, space_group, seed) combination: builds a random
crystal via pyxtal().from_random(), converts to a supercell sized by
balanced_supercell (aspect-ratio-minimizing multiplicity search), rattles
site positions, and writes the result as
output_root/<formula>/<space_group>/<seed>/POSCAR.

Many (composition, space_group, seed) combinations are not realizable --
pyxtal.from_random sets `.valid = False` (and can also raise outright for
some clearly-incompatible inputs) rather than guaranteeing success. Both
are caught per-attempt so one bad combination doesn't abort the whole
batch; a summary of successes/skips prints at the end rather than failing
silently or crashing partway through a large generation run.

CONFIRMED (against the live Pd-Se pipeline, not a theoretical concern): an
extreme-ratio/high-symmetry combination (e.g. a near-single-species
composition against a high-numbered space group) can make pyxtal's own
internal Wyckoff-combination search run indefinitely rather than raising
or returning invalid. A SIGALRM-based wall-clock timeout was tried first
and confirmed NOT sufficient: the exact same combination still stalled the
batch at an identical attempt count across repeated runs, despite the
timeout firing -- pyxtal's own internal retry loop swallows the
alarm-raised exception rather than letting it propagate (a signal can't
force-interrupt code that catches everything around its own retry loop).
Each attempt now runs in its own subprocess instead (multiprocessing,
"fork" context) -- a hard terminate()/kill() from the parent works
regardless of what the child's own exception handling does, since nothing
inside the child can intercept an external SIGTERM/SIGKILL. This also
isolates the parent from a hard crash (e.g. a segfault in pyxtal/spglib
for a sufficiently degenerate input) in any one attempt.
"""
import multiprocessing as mp
import os
from itertools import product

import numpy as np
from ase.io import write
from pyxtal import pyxtal
from tqdm import tqdm


def balanced_supercell(atoms, max_atoms=100, max_mult=6, score_std=False):
    """
    Picks the (nx, ny, nz) repeat under max_mult that stays within
    max_atoms and minimizes cell-length aspect ratio (or std, if
    score_std), so generated supercells aren't pathologically long/thin.
    """
    natoms = len(atoms)
    cell = atoms.get_cell()
    lengths = np.linalg.norm(cell, axis=1)

    best_score = np.inf
    best_mult = None

    for nx, ny, nz in product(range(1, max_mult + 1), range(1, max_mult + 1), range(1, max_mult + 1)):
        mult_atoms = natoms * nx * ny * nz
        if mult_atoms > max_atoms:
            continue

        new_lengths = lengths * np.array([nx, ny, nz])
        score = np.std(new_lengths) if score_std else max(new_lengths) / min(new_lengths)

        if score < best_score and (nx, ny, nz) != (1, 1, 1):
            best_score = score
            best_mult = (nx, ny, nz)

    if best_mult is None:
        return atoms.copy()
    return atoms.repeat(best_mult)


def _formula_label(species, counts):
    """e.g. species=('Pd','Se'), counts=[2,3] -> 'Pd2Se3'."""
    return "".join(f"{sym}{n}" for sym, n in zip(species, counts))


def _generate_one_worker(dimensionality, sg, species, counts, seed,
                          max_atoms, max_mult, rattle_stdev, out_dir, queue):
    """
    Runs entirely inside a child process (see this module's own docstring
    for why) -- does the full per-attempt work (generate, supercell,
    rattle, write POSCAR) and reports back ("ok", None) or
    ("error", <reason str>) via `queue`, rather than sending the generated
    Atoms object back to the parent (simpler than round-tripping a pickled
    Atoms through the queue, and keeps the parent's job purely supervisory).
    """
    try:
        crystal = pyxtal()
        crystal.from_random(dimensionality, sg, list(species), list(counts), random_state=seed)
        if not crystal.valid:
            queue.put(("error", "pyxtal reported an invalid structure"))
            return

        atoms = crystal.to_ase()
        atoms = balanced_supercell(atoms, max_atoms=max_atoms, max_mult=max_mult)
        atoms.rattle(stdev=rattle_stdev, seed=seed)

        os.makedirs(out_dir, exist_ok=True)
        write(os.path.join(out_dir, "POSCAR"), atoms)
        queue.put(("ok", None))
    except Exception as e:
        queue.put(("error", str(e)))


def generate_pyxtal_structures(output_root, compositions, space_groups, seeds,
                                dimensionality=3, max_atoms=100, max_mult=6,
                                rattle_stdev=0.1, species=("Pd", "Se"),
                                per_attempt_timeout_seconds=60):
    """
    compositions: list of [n_A, n_B] atom-count pairs -- pyxtal's own
    numIons convention (NOT reduced ratios; pyxtal multiplies as needed to
    satisfy the target space group's Wyckoff-site constraints).
    space_groups: list of international space-group numbers (1-230) to
    attempt for every composition.
    seeds: list of ints -- one generation attempt per
    (composition, space_group, seed) combination, each independently
    reproducible.
    per_attempt_timeout_seconds: hard wall-clock cap on each attempt's
    child process (set to 0/None to disable -- see this module's own
    docstring for why this is a subprocess kill, not a signal/try-except).

    Prints a live tqdm progress bar (current formula/space-group/seed plus
    running success/skip/timeout counts) across the full
    compositions x space_groups x seeds product, so a long sweep's actual
    progress -- vs. a genuine hang -- is visible in real time rather than
    only inferable from a final summary. Returns
    {"n_success": int, "n_skipped": int} and prints up to 20 skip reasons
    at the end.
    """
    os.makedirs(output_root, exist_ok=True)

    n_success, n_skipped, n_timeout = 0, 0, 0
    skipped = []

    ctx = mp.get_context("fork")
    combos = list(product(compositions, space_groups, seeds))

    n_already_done = 0
    progress = tqdm(combos, desc="pyxtal generation", unit="attempt")
    for counts, sg, seed in progress:
        formula = _formula_label(species, counts)
        out_dir = os.path.join(output_root, formula, str(sg), str(seed))

        # Resumability: a restarted sweep (e.g. after killing a stalled
        # run) shouldn't redo already-written attempts -- also means a
        # combination that stalls again lands at a NEW attempt, not the
        # same already-completed one, so a repeat kill actually makes
        # progress past the previous stall point.
        if os.path.exists(os.path.join(out_dir, "POSCAR")):
            n_success += 1
            n_already_done += 1
            progress.set_postfix(ok=n_success, skipped=n_skipped, timeout=n_timeout, resumed=n_already_done)
            continue

        queue = ctx.Queue()
        proc = ctx.Process(
            target=_generate_one_worker,
            args=(dimensionality, sg, species, counts, seed,
                  max_atoms, max_mult, rattle_stdev, out_dir, queue),
        )
        proc.start()
        proc.join(timeout=per_attempt_timeout_seconds or None)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            n_timeout += 1
            n_skipped += 1
            skipped.append((formula, sg, seed, f"timed out after {per_attempt_timeout_seconds}s (subprocess killed)"))
        elif queue.empty():
            n_skipped += 1
            skipped.append((formula, sg, seed, f"subprocess exited (code {proc.exitcode}) with no result -- likely crashed"))
        else:
            status, payload = queue.get()
            if status == "ok":
                n_success += 1
            else:
                n_skipped += 1
                skipped.append((formula, sg, seed, payload))

        queue.close()
        progress.set_postfix(ok=n_success, skipped=n_skipped, timeout=n_timeout)

    print(f"generate_pyxtal_structures: wrote {n_success} structure(s) "
          f"({n_already_done} already present from a prior run), skipped {n_skipped} "
          f"(of which {n_timeout} timed out)")
    if skipped:
        print("Skipped (composition, space_group, seed, reason):")
        for formula, sg, seed, reason in skipped[:20]:
            print(f"  {formula}, sg={sg}, seed={seed}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return {"n_success": n_success, "n_skipped": n_skipped}
