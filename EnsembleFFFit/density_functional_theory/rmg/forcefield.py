import os
import re
import xml.etree.ElementTree as ET


class Forcefield:
    def __init__(self, forcefield_xml_path):
        self.forcefield_xml_path = forcefield_xml_path
        self.force = False
        self.force_convergent = False
        self.scf = False
        self.scf_convergent = False
        self.parse_convergence()

    def parse_convergence(self):
        if not os.path.exists(self.forcefield_xml_path):
            return
        with open(self.forcefield_xml_path) as f:
            xml = f.read()
        try:
            xml_root = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")
            for group in xml_root.findall('converged'):
                self.force = eval(group.find('force').text)
                self.force_convergent = eval(group.find('force_convergent').text)
                self.scf = eval(group.find('scf').text)
                self.scf_convergent = eval(group.find('scf_convergent').text)
        except ET.ParseError:
            pass
