/* LD_PRELOAD shim: initialise HIP before the program's main() -- and so
 * before MPI_Init. Inside the container, a Cray MPICH MPI_Init in a job
 * spanning more than one node leaves HIP unable to find any GPU
 * (hipGetDeviceCount -> 0, GTL hsa_init -> HSA_STATUS_ERROR_OUT_OF_RESOURCES)
 * unless HIP was initialised first (INVESTIGATION_NOTES.md, Phase 2). This
 * lets rmg-gpu run unmodified:
 *   LD_PRELOAD=/path/to/libhip_preinit.so rmg-gpu rmg_input
 * Set HIP_PREINIT_VERBOSE=1 to print the device count each process saw. */
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((constructor)) static void hip_preinit(void) {
    int n = 0;
    hipError_t err = hipInit(0);
    if (err == hipSuccess) err = hipGetDeviceCount(&n);
    if (getenv("HIP_PREINIT_VERBOSE"))
        fprintf(stderr, "hip_preinit: %s, %d device(s)\n", hipGetErrorString(err), n);
}
