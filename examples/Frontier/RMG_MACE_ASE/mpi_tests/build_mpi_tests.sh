#!/usr/bin/bash
# Build mpi_hello natively with the same module stack as rmg-gpu (README.md
# step 3), so it links the same Cray MPICH / PMI / GTL libraries and is run
# inside the container with the same RMG_LD_LIBRARY_PATH.
set -euo pipefail
cd "$(dirname "$0")"
module load PrgEnv-gnu/8.6.0 gcc-native/13.2 craype-x86-milan craype-accel-amd-gfx90a rocm/6.3.1
for variant in mpi_hello mpi_hello_hipfirst; do
    flags=""; [[ $variant == mpi_hello_hipfirst ]] && flags="-DHIP_FIRST"
    cc -O2 $flags -o $variant.new mpi_hello.c -I"${ROCM_PATH}/include" -D__HIP_PLATFORM_AMD__ -L"${ROCM_PATH}/lib" -lamdhip64
    mv -f $variant.new $variant   # rename, so a job already running the old binary is unaffected
done
cc -O2 -shared -fPIC -o libhip_preinit.so.new hip_preinit.c -I"${ROCM_PATH}/include" -D__HIP_PLATFORM_AMD__ \
    -L"${ROCM_PATH}/lib" -lamdhip64
mv -f libhip_preinit.so.new libhip_preinit.so
ldd mpi_hello | grep -E "mpi|pmi|gtl|amdhip" 
