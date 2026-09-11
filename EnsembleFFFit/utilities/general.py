import importlib.util
import ast
import os
import re
import shutil
import sys
from pathlib import Path


def copy_and_transform_files(source_directory, target_directory, pattern, dest_path_fn, transform_fn=shutil.copy2):
    """
    Walk `source_directory`; for every file matching `pattern`, compute its
    destination via `dest_path_fn(source_path, match)` and produce it via
    `transform_fn(source_path, dest_path)` (default: a plain copy). Splits
    "where does this file's copy go" from "how do the bytes get there" so the
    same walk/match logic can be reused for a rename-only copy, a format
    conversion, or any other per-stage transform (e.g. fitted force-field
    models moving to an MD stage's inputs_directory, MD structures moving to a
    DFT stage's inputs_directory, LAMMPS dump reformatting, etc.).
    """
    if not os.path.isdir(source_directory):
        raise FileNotFoundError(f"Source directory '{source_directory}' does not exist.")

    file_pattern = re.compile(pattern)

    for root, _, files in os.walk(source_directory):
        for f in files:
            match = file_pattern.match(f)
            if not match:
                continue

            source_path = Path(root) / f
            dest_path = dest_path_fn(source_path, match)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                transform_fn(source_path, dest_path)
                print(f"Wrote '{source_path}' -> '{dest_path}'")
            except Exception as e:
                print(f"Failed to write '{source_path}' -> '{dest_path}': {e}")


def make_mirrored_rename_dest_path_fn(source_directory, target_directory, target_name):
    """
    Return a `dest_path_fn` that mirrors `source_path`'s parent directory
    (relative to `source_directory`) under `target_directory`, renaming the
    file to `target_name`. Covers e.g. the MACE case: `MACE_i.model` anywhere
    under `source_directory` becomes `target_directory/<same subtree>/model.model`.
    """
    source_directory = Path(source_directory)
    target_directory = Path(target_directory)

    def dest_path_fn(source_path, match):
        rel_dir = source_path.parent.relative_to(source_directory)
        return target_directory / rel_dir / target_name

    return dest_path_fn


def ensemble_fffit_pythonpath():
    """
    Repo root containing the EnsembleFFFit package, derived from this module's
    own location so it always matches whatever install is actually running
    this code -- not a hardcoded path string. Meant to be passed into a
    chore's `Resources.env['PYTHONPATH']` so EnsembleFFFit is importable
    inside the chore's own process, regardless of what PYTHONPATH the
    submitting shell's ambient environment happens to carry at submission
    time (e.g. host-module state leaked through Apptainer without
    --cleanenv, from wherever Flux itself was started).
    """
    utilities_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(utilities_dir)
    return os.path.dirname(package_dir)


def import_module_from_path(module_name, path):
    """
    Import a user-authored driver script (e.g. an MD execution script) by its
    full path -- it lives alongside a run's inputs_directory rather than
    inside the installed EnsembleFFFit package, so a plain `import` won't
    find it.

    Registers the module in sys.modules under module_name BEFORE
    executing it -- required (not cosmetic), confirmed via a real failure
    (2026-09): a driver script that uses multiprocessing.Pool internally
    (e.g. MD/finite_temperature/coordination_check/cn_checker.py) needs
    its own functions to be re-importable by name in each worker process
    so they can be unpickled; without this, pickling a function defined
    in this module raises "PicklingError: import of module '<name>'
    failed" the moment any dynamically-loaded driver tries to hand a
    function to a Pool. Matches the standard importlib recipe for
    executing a spec-loaded module (see importlib's own docs), just not
    followed here previously since no prior driver script happened to use
    multiprocessing.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def mirror_completed_leaves(source_root, dest_root, filenames):
    """
    For every leaf directory under source_root containing ALL of
    `filenames`, copy each into the mirrored location under dest_root --
    e.g. mirroring completed DFT validation results (POSCAR +
    properties.json) into an MD stage's inputs_directory so its single
    points have a ground-truth pair to compare against. Skips any leaf
    missing even one of `filenames` (a run that hasn't converged/finished
    yet). Returns the list of source leaf directories actually copied.
    """
    dest_path_fns = {name: make_mirrored_rename_dest_path_fn(source_root, dest_root, name) for name in filenames}

    copied = []
    for dirpath, _, files in os.walk(source_root):
        if not all(name in files for name in filenames):
            continue
        for name in filenames:
            src = Path(dirpath) / name
            dst = dest_path_fns[name](src, None)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied.append(dirpath)

    return copied


def unpack_trajectory_frames(source_root, dest_root, traj_filename="md_run.traj"):
    """
    For every `traj_filename` found under source_root (e.g. finite-
    temperature MD trajectories), write each of its frames as
    dest_root/<relpath-of-run>/<frame_index>/POSCAR -- one subtree per run,
    each run's own frames numbered 0..N-1, matching the (md_name, md_image)
    convention EnsembleFFFit.analysis.dict_parsers/variance expect. Returns
    (num_runs, num_frames_written). Imports ase.io lazily so importing this
    module doesn't require ase to be installed for callers that never use
    this function.
    """
    from ase.io import read, write

    num_runs = 0
    num_frames = 0

    for dirpath, _, files in os.walk(source_root):
        if traj_filename not in files:
            continue

        rel_run = os.path.relpath(dirpath, source_root)
        frames = read(os.path.join(dirpath, traj_filename), index=":")

        for i, atoms in enumerate(frames):
            out_dir = os.path.join(dest_root, rel_run, str(i))
            os.makedirs(out_dir, exist_ok=True)
            write(os.path.join(out_dir, "POSCAR"), atoms)
            num_frames += 1

        num_runs += 1

    return num_runs, num_frames


def print_and_write(lines, path, header=""):
    """
    Print `lines` (already-formatted strings) joined by newlines, and write
    the same text to `path` (creating parent directories as needed) --
    keeps a script's stdout output and its on-disk record (e.g. a ranking
    or selection manifest) identical by construction rather than by
    hand-keeping two copies in sync.
    """
    text = (header + "\n\n" if header else "") + "\n".join(lines) + "\n"
    print(text)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

    return path


def parse_list(arg):
    """
    Try to parse `arg` as a Python literal list via ast.literal_eval.
    If that fails, fall back to simple comma-splitting.
    """
    try:
        val = ast.literal_eval(arg)
        if isinstance(val, list):
            return val
        # if it parsed to something else, keep going to split
    except (ValueError, SyntaxError):
        pass
    # fallback
    return arg.split(',')
