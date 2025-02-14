# Ensemble-FF-Fit
Wrapper to fit reactive an ensemble of force-fields to ab initio data using adaptive job scheduling, with UQ on-the-fly to generate new training data to improve the force-fields. 
PyRMG code allows performing high-throughput ab initio DFT calculations using the RMG code (https://github.com/RMGDFT/rmgdft).  MatEnsemble is used to perform adaptive asynchronous job scheduling.  

Currently force-fields implemented: JAX-ReaxFF and SNAP.  
