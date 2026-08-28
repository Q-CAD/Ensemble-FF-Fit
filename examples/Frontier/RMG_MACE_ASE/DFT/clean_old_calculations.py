#!/usr/bin/env python3
"""
Walk a directory tree looking for subdirectories that contain either a
"POSCAR" or an "unused_POSCAR" file.

- If "POSCAR" is present: remove every other file in that subdirectory
  except "POSCAR" and "METADATA".
- If "unused_POSCAR" is present (and "POSCAR" is not): rename it to
  "POSCAR", then remove every other file except "POSCAR" and "METADATA".

Subdirectories with neither file are left untouched.

By default this runs in dry-run mode and only prints what it would do.
Pass --execute to actually perform the rename/delete operations.
"""

import argparse
import sys
from pathlib import Path

KEEP_NAMES = {"POSCAR", "METADATA"}


def process_directory(directory: Path, execute: bool) -> None:
    has_poscar = (directory / "POSCAR").is_file()
    has_unused = (directory / "unused_POSCAR").is_file()

    if not has_poscar and not has_unused:
        return

    if has_unused and not has_poscar:
        src = directory / "unused_POSCAR"
        dst = directory / "POSCAR"
        print(f"[rename] {src} -> {dst}")
        if execute:
            src.rename(dst)
    elif has_unused and has_poscar:
        print(
            f"[skip-rename] {directory} has both POSCAR and unused_POSCAR; "
            f"leaving unused_POSCAR as-is (it will be deleted below)"
        )

    for item in directory.iterdir():
        if item.is_file() and item.name not in KEEP_NAMES:
            print(f"[delete] {item}")
            if execute:
                item.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_directory",
        type=str,
        help="Top-level directory to search recursively.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the rename/delete operations. "
        "Without this flag, only prints what would happen.",
    )
    args = parser.parse_args()

    root = Path(args.root_directory)
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not args.execute:
        print("DRY RUN -- no files will be modified. Pass --execute to apply changes.\n")

    for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
        process_directory(directory, args.execute)

    # Also check the root directory itself, not just its subdirectories.
    process_directory(root, args.execute)


if __name__ == "__main__":
    main()
