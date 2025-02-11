# `parse2fit`

![alt text](Ensemble-FF-fit.jpg?raw=true)

`parse2fit` parses DFT calculations into input files for force field training. It currently supports VASP and RMG DFT codes and the Reax force field.

Future updates will add support for parsing raw files into intermediate data types. 

# Setup

1. Clone `parse2fit` locally:

        git clone https://github.com/rymo1354/Ensemble-FF-fit.git

2. Build the base conda environment:

        conda env create -f Ensemble-FF-fit.yml

3. Install `parse2fit` in the conda environment:

        python setup.py install

# Code Design

`parse2fit` employs `Parser()` classes (e.g., for VASP or RMG) to convert raw data into `Property()` objects (e.g., `Energy()`, `Charge()`, `Force()`, etc.). 

`Entries()` (e.g., `ReaxEntry()`) are built from `Property()` classes, and contain logic to transform raw data into input files, including unit conversions and relative energy calculations for Reax.   

`ParserFactory()` automatically constructs objects based on detected files (e.g., `vasprun.xml` or `forcefield.xml`) in the directory passed. 

The `ReadWrite()` classes use `.yml` instructions to generate the appropriate `Property()` and `Entry()` objects and write the corresponding input files for force field training. Below is an example for `ReaxRW()`. 

# Usage
The following code generates input files as specified by a configuration `.yml` file.

    config_path = '/path/to/your/configuration.yml'
    rw = ReadWriteFactory(config_path=config_path).get_writer()
    rw.write_trainsetins()

Descriptions of supported `.yml` configurations are provided below. 

## Reax

Reax force field fitting minimizes an objective function built from structures included in a `geo` file and training data included in a `trainset.in` file. Adjusting data weights or included data can significantly affect the fit and performance of the force field. The Reax `.yml` format supports data sampling and weighting for flexibility.

### 1. `output_format`:

(str) Specifies the `.yml` format. Currently, only `jax-reaxff` is supported.

### 2. `generation_parameters`:
    
`runs_to_generate`: (int): Number of unique input combinations to generate.

`input_directory`: (str) Path to input `geo` and `trainset.in` files.

`fix_input_weights`: (bool): If True, weights in `trainset.in` are fixed. If False, weights can change during generation.

`output_directory`: (str): Path to write new Reax input files. 

### 3. `default_weights`:

Allows for automated weighting of data points. Supported schemes include:

1. Binary: Assigns `min` or `max` weights randomly, with probabilities defined by `split`.

2. Magnitude: Weights are exponential values within the range `[min, max]`, and are scaled by `kT` and `scale`.

3. Normal: Distributes weights normally between `min` and `max`, with spread controlled by `sigma` and `scale`.

### 4. `input_paths`:

Defines the data to be sampled and its weighting scheme:

Directory paths: Specify absolute paths to DFT run directories or directory trees.

Run types: Type of MD calculation to be performed during force field fitting. 

Data types: Select properties (e.g., energy, forces, charges) to parse.

Weights: Optionally override default_weights for specific data.

Energy references: Specify reference states for subtracting energies (e.g., for formation energy).

Example entry:

    /path/to/directory_tree_of_DFT_runs:
        structure:
            rutype: 'NORMAL RUN'
        energy:
            subtract:
                - /path/to/reference1_directory
                - /path/to/reference2_directory
            get_divisors: True 
        forces: True
        charges:
            weights:
                type: 'binary'
                min: 0
                max: 1
                split: 0.85
        angles: False

Energies are written to `trainset.in` without duplicates, prioritizing shorter or named directory paths.



