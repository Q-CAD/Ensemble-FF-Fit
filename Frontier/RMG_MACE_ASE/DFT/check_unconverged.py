from pathlib import Path

KEEP_NAMES = {"rmg_input", "METADATA", "POSCAR"}
EXECUTE = False  # flip to True once you've reviewed the dry-run output

for d in Path('.').rglob('*'):
    if d.is_dir() and (d / 'rmg_input').is_file() and not (d / 'properties.json').is_file():
        print(d)
        for item in d.iterdir():
            if item.is_file() and item.name not in KEEP_NAMES:
                print(f"  [delete] {item}")
                if EXECUTE:
                    item.unlink()
