from abc import ABC, abstractmethod
from pathlib import Path
import os
import json

from pymatgen.core import Structure
from pymatgen.io.vasp import Vasprun


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

        for root, _, _ in os.walk(self.directory_to_parse):
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

        for root, _, _ in os.walk(self.directory_to_parse):
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
