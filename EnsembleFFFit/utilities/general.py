import importlib.util
import ast
import os
import re
import shutil
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
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
