from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
import os
import json

from pymatgen.core import Structure
from pymatgen.io.vasp import Vasprun
from tqdm import tqdm


class DirectoryParser(ABC):
    def __init__(self, directory_to_parse):
        """
        Store the root directory to walk when parsing.
        """
        self.directory_to_parse = Path(directory_to_parse)

    @abstractmethod
    def parse_directory(self, label_tuple):
        """
        Walk `directory_to_parse` and return a nested dict of
        {label: {run: {image: properties}}}.
        """
        pass

    @abstractmethod
    def existence_check(self, root: Path):
        """
        Check whether `root` contains the expected files for this parser;
        return the path(s) if so, else raise.
        """
        pass

    @abstractmethod
    def get_property_values(self, *paths):
        """
        Extract (energy, fx, fy, fz, structure) from the given path(s).
        """
        pass

    def _nested_set(self, dct, keys, value):
        """
        Create nested dictionary structure and assign value.
        """
        cur = dct
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = value

    def naming_convention(self, path, label_tuple):
        """
        Split a path's parts into (label, run, image) using `label_tuple` as the
        slice bounds for the label segment; `run` is everything between the label
        and the final path component, `image` is the final component.
        """
        parts = Path(path).parts
        label = "/".join(parts[label_tuple[0]:label_tuple[1]])
        run = "/".join(parts[label_tuple[1]:-1])
        image = parts[-1]
        return label, run, image


class ASEParser(DirectoryParser):
    def existence_check(self, root: Path):
        """
        Return (properties.json path, POSCAR path) if both exist under root,
        else raise ValueError.
        """
        properties_path = root / "properties.json"
        poscar_path = root / "POSCAR"

        if properties_path.exists() and poscar_path.exists():
            return properties_path, poscar_path
        raise ValueError

    def get_property_values(self, properties_path, poscar_path):
        """
        Load energy/forces from properties.json and structure from POSCAR.
        """
        with open(properties_path) as f:
            data = json.load(f)

        structure = Structure.from_file(poscar_path)

        return (
            data["energy"],
            data["fx"],
            data["fy"],
            data["fz"],
            structure,
        )

    def parse_directory(self, label_tuple):
        """
        Walk the directory tree applying `naming_convention`, `existence_check`,
        and `get_property_values` to build the nested properties dict, skipping
        entries where `existence_check` raises ValueError.
        """
        full_dct = {}

        for root, _, _ in tqdm(os.walk(self.directory_to_parse),
                                desc=f"Parsing {self.directory_to_parse}", unit="dir"):
            root = Path(root)

            try:
                props, poscar = self.existence_check(root)
                energy, fx, fy, fz, structure = self.get_property_values(props, poscar)
            except ValueError:
                continue

            label, run, image = self.naming_convention(root, label_tuple)

            self._nested_set(
                full_dct,
                [label, run, image],
                {
                    "energy": energy,
                    "structure": structure,
                    "fx": fx,
                    "fy": fy,
                    "fz": fz,
                },
            )

        return full_dct


class VASPParser(DirectoryParser):
    def existence_check(self, root: Path):
        """
        Return vasprun.xml path if it exists under root, else raise ValueError.
        """
        vasprun_path = root / "vasprun.xml"
        if vasprun_path.exists():
            return vasprun_path
        raise ValueError

    def get_property_values(self, vasprun_path):
        """
        Load energy/forces/structure from a VASP vasprun.xml's final ionic step.
        Assumes `ionic_steps[-1]['forces']` is a numpy array.
        """
        v = Vasprun(vasprun_path)

        forces = v.ionic_steps[-1]["forces"]
        structure = v.structures[-1]

        return (
            v.final_energy,
            forces[:, 0],
            forces[:, 1],
            forces[:, 2],
            structure,
        )

    def parse_directory(self, label_tuple):
        """
        Walk the directory tree applying `naming_convention`, `existence_check`,
        and `get_property_values` to build the nested properties dict.
        """
        full_dct = {}

        for root, _, _ in tqdm(os.walk(self.directory_to_parse),
                                desc=f"Parsing {self.directory_to_parse}", unit="dir"):
            root = Path(root)

            try:
                vasprun_path = self.existence_check(root)
                energy, fx, fy, fz, structure = self.get_property_values(vasprun_path)
            except ValueError:
                continue

            label, run, image = self.naming_convention(root, label_tuple)

            self._nested_set(
                full_dct,
                [label, run, image],
                {
                    "energy": energy,
                    "structure": structure,
                    "fx": fx,
                    "fy": fy,
                    "fz": fz,
                },
            )

        return full_dct


class PropertiesOnlyParser(DirectoryParser):
    """
    Like ASEParser, but requires only properties.json -- no co-located
    structure file. Needed for MDMatEnsemble-style single-point output
    (e.g. reaxff_validation_single_points): its run_directory/
    inputs_directory split means a chore's own output leaf
    (run_directory/<...>/properties.json) never has a structure file
    sitting next to it at all -- the structure.lmp/POSCAR lives entirely
    separately, under inputs_directory. CONFIRMED (2026-09) as a real bug
    ASEParser hit here, not theoretical: its existence_check requiring
    both properties.json AND POSCAR silently skipped every single leaf
    under such a tree, since POSCAR is never present there, producing a
    fully empty parsed dict with no error -- exactly the shape of failure
    that later crashed downselect_force_fields.select_and_copy with
    "IndexError: list index out of range" (an empty label list, indexed
    into as if it had entries).

    get_property_values returns structure=None always -- callers that
    need real per-image Structure objects should use ASEParser (or
    VASPParser) instead; this parser is for energy/force-only consumers
    (e.g. best_force_field.rank_force_fields_combined).
    """
    def existence_check(self, root: Path):
        """Return the properties.json path if it exists under root, else raise ValueError."""
        properties_path = root / "properties.json"
        if properties_path.exists():
            return properties_path
        raise ValueError

    def get_property_values(self, properties_path):
        """Load energy/forces from properties.json; structure is always None."""
        with open(properties_path) as f:
            data = json.load(f)
        return (
            data["energy"],
            data["fx"],
            data["fy"],
            data["fz"],
            None,
        )

    def parse_directory(self, label_tuple):
        """Walk the directory tree applying `naming_convention`, `existence_check`,
        and `get_property_values` to build the nested properties dict, skipping
        entries where `existence_check` raises ValueError."""
        full_dct = {}

        for root, _, _ in tqdm(os.walk(self.directory_to_parse),
                                desc=f"Parsing {self.directory_to_parse}", unit="dir"):
            root = Path(root)

            try:
                properties_path = self.existence_check(root)
                energy, fx, fy, fz, structure = self.get_property_values(properties_path)
            except ValueError:
                continue

            label, run, image = self.naming_convention(root, label_tuple)

            self._nested_set(
                full_dct,
                [label, run, image],
                {
                    "energy": energy,
                    "structure": structure,
                    "fx": fx,
                    "fy": fy,
                    "fz": fz,
                },
            )

        return full_dct


def _parse_one_label_subtree(args):
    """
    Pool worker: parse ONE immediate child directory of a labeled tree's
    root (one force-field variant's own subtree) -- see
    parse_labeled_tree's own docstring for why this is a safe,
    embarrassingly-parallel split (same reasoning as
    relative_energy_comparison.py's own per-run multiprocessing: every
    label's subtree is independent file I/O, no shared state, and
    label_tuple is computed from the ORIGINAL root -- not recomputed
    relative to this child -- so naming_convention still slices out the
    same label/run/image parts it would have from one big sequential
    walk over the whole tree).
    """
    parser_cls, child_dir, label_tuple = args
    return parser_cls(child_dir).parse_directory(label_tuple=label_tuple)


def parse_labeled_tree(root, parser_cls=ASEParser, num_processes=None):
    """
    Parse `root` treating the immediate child directory name as the label
    -- e.g. a force-field variant number/name -- giving
    {label: {run: {image: props}}}. label_tuple is computed from root's own
    resolved depth, not hardcoded, so this works regardless of where the
    tree lives on disk.

    num_processes (optional, default None i.e. sequential -- the original,
    unchanged behavior): if set >1, each immediate child of root (one
    label's own subtree) is parsed in its own multiprocessing.Pool worker
    instead of one single sequential os.walk over the entire tree. Added
    (2026-10) once this became the real bottleneck for large ensembles --
    the dominant cost is per-structure file I/O (properties.json/POSCAR
    reads), repeated once per label, so this scales near-linearly with
    available cores. Each label contributes its own, non-colliding
    top-level key, so merging results back together is a plain dict
    update, not a deep merge.
    """
    root = str(Path(root).resolve())
    base_depth = len(Path(root).parts)
    label_tuple = (base_depth, base_depth + 1)

    if not num_processes or num_processes <= 1:
        return parser_cls(root).parse_directory(label_tuple=label_tuple)

    children = sorted(p for p in Path(root).iterdir() if p.is_dir())
    if not children:
        return {}
    tasks = [(parser_cls, str(child), label_tuple) for child in children]

    full_dct = {}
    with Pool(processes=min(num_processes, len(tasks))) as pool:
        for result in tqdm(pool.imap_unordered(_parse_one_label_subtree, tasks),
                            total=len(tasks), desc=f"Parsing {root}", unit="label"):
            full_dct.update(result)
    return full_dct


def parse_reference_tree(root, parser_cls=ASEParser):
    """
    Parse `root` with no extra label level -- {run: {image: props}} --
    e.g. a DFT ground-truth mirror with no per-variant subdirectory of its
    own. Unwraps the single "" label parse_directory produces in this case,
    so callers get the run/image tree directly rather than needing to know
    about the empty-label implementation detail.
    """
    root = str(Path(root).resolve())
    base_depth = len(Path(root).parts)
    return parser_cls(root).parse_directory(label_tuple=(base_depth, base_depth)).get("", {})
