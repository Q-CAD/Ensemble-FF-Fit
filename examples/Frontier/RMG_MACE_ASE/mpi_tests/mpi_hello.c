/* Minimal check that N processes launched by Flux (or srun) join ONE MPI
 * job, and which GPU/PMI library each rank ends up with. Built against the
 * same Cray MPICH as rmg-gpu (build_mpi_tests.sh). Prints one line per rank:
 *   rank r/size host=<h> local=<l> gpu_count=<n> gpu0_pci=<bus> ROCR=.. CUDA=.. pmi=<libpmi path>
 * then a host-buffer Allreduce check, and -- with argument "gpu" and
 * MPICH_GPU_SUPPORT_ENABLED=1 -- the same Allreduce on hipMalloc'd buffers.
 * size=1 on every rank means each process came up as its own singleton.
 * Built with -DHIP_FIRST (mpi_hello_hipfirst), the GPUs are queried before
 * MPI_Init instead of after it -- to tell whether MPI_Init itself (e.g.
 * libfabric's GPU-memory setup when the job spans nodes) breaks HIP. */
#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void loaded_pmi(char *out, size_t n) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[1024];
    snprintf(out, n, "none");
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        char *p = strstr(line, "/");
        if (p && strstr(p, "libpmi")) {
            p[strcspn(p, "\n")] = 0;
            snprintf(out, n, "%s", p);
            break;
        }
    }
    fclose(f);
}

int main(int argc, char **argv) {
    int rank, size, local = -1, ndev = 0;
    char host[256], pmi[512], pci[64] = "-";
#ifdef HIP_FIRST
    hipError_t pre = hipGetDeviceCount(&ndev);
    if (pre != hipSuccess) ndev = 0;
#endif
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local);
    gethostname(host, sizeof host);
    loaded_pmi(pmi, sizeof pmi);
#ifndef HIP_FIRST
    if (hipGetDeviceCount(&ndev) != hipSuccess) ndev = 0;
#endif
    if (ndev > 0) {
        hipSetDevice(local % ndev);
        hipDeviceGetPCIBusId(pci, sizeof pci, local % ndev);
    }
    const char *rocr = getenv("ROCR_VISIBLE_DEVICES"), *cuda = getenv("CUDA_VISIBLE_DEVICES");
    printf("rank %d/%d host=%s local=%d gpu_count=%d gpu_pci=%s ROCR=%s CUDA=%s pmi=%s\n", rank, size, host,
           local, ndev, pci, rocr ? rocr : "-", cuda ? cuda : "-", pmi);
    fflush(stdout);

    double x = rank + 1, sum = 0;
    MPI_Allreduce(&x, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("host allreduce: %s (got %g, want %g)\n",
                          sum == size * (size + 1) / 2.0 ? "OK" : "WRONG", sum, size * (size + 1) / 2.0);

    if (argc > 1 && strcmp(argv[1], "gpu") == 0 && ndev > 0) {
        double *d, *dsum, h = 0;
        hipMalloc((void **)&d, sizeof(double));
        hipMalloc((void **)&dsum, sizeof(double));
        hipMemcpy(d, &x, sizeof(double), hipMemcpyHostToDevice);
        MPI_Allreduce(d, dsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        hipMemcpy(&h, dsum, sizeof(double), hipMemcpyDeviceToHost);
        if (rank == 0) printf("gpu-buffer allreduce: %s (got %g)\n", h == size * (size + 1) / 2.0 ? "OK" : "WRONG", h);
    }
    MPI_Finalize();
    return 0;
}
