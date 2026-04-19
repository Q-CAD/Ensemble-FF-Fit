# Ensemble-FF-Fit
![alt text](Ensemble-FF-Fit.png?raw=true)

This package allows **data** and **time efficient fine-tuning of physics-based as well as machine-learning interatomic potentials (MLIPS)**, including universal MLIPs, by using an ensemble approach to fit MLIPs to ab initio data using adaptive asynchronous job scheduling while incorporating UQ on-the-fly. to generate new training data to improve the force-fields. 

PyRMG code allows performing high-throughput ab initio DFT calculations using the RMG code (https://github.com/RMGDFT/rmgdft).  MatEnsemble is used to perform adaptive asynchronous job scheduling.  

Currently the package is implemented for: JAX-ReaxFF and MACE type force-fields.

Future plans: Include support for universal ML-FFs such as CHGNET and M3GNet for quantum materials.  
