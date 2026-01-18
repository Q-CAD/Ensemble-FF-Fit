import sys
import os
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    dir_with_vaspruns = sys.argv[1]
    vasprun_path = os.path.join(dir_with_vaspruns, 'vasprun.xml')

    tree = ET.parse(vasprun_path)
    root = tree.getroot()

    # Find all calculation elements (each corresponds to one MD step)
    calculations = root.findall("calculation")

    for i, calc in enumerate(calculations):
        # Create a new root node
        new_root = ET.Element(root.tag, root.attrib)

        # Copy header/global elements except 'calculation'
        for child in root:
            if child.tag != "calculation":
                new_root.append(child)

        # Add only the i-th calculation element
        new_root.append(calc)

        # (Optional) sanity check: extract the structure for this step
        structure_elem = calc.find("structure")
        if structure_elem is None:
            print(f"Warning: No structure found in step {i}")
        else:
            natoms = len(structure_elem.find("varray[@name='positions']"))
            print(f"Step {i}: Found structure with {natoms} atoms")

        # Write the single-step vasprun.xml
        write_to = os.path.join(dir_with_vaspruns, 'single_image', str(i))
        os.makedirs(write_to, exist_ok=True)
        new_tree = ET.ElementTree(new_root)
        new_tree.write(os.path.join(write_to, "vasprun.xml"), encoding="utf-8", xml_declaration=True)


