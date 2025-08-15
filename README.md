# Ensemble-FF-Fit
![alt text](Ensemble-FF-Fit.png?raw=true)

Wrapper to fit reactive an ensemble of force-fields to ab initio data using adaptive job scheduling, with UQ on-the-fly to generate new training data to improve the force-fields. 
PyRMG code allows performing high-throughput ab initio DFT calculations using the RMG code (https://github.com/RMGDFT/rmgdft).  MatEnsemble is used to perform adaptive asynchronous job scheduling.  

Currently wrappers implemented for: JAX-ReaxFF and SNAP force-fields.

Future plans: Include support for universal ML-FFs such as MACE, CHGNET and M3GNet for quantum materials.  
