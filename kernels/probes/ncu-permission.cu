// One tiny launch to check hardware-counter access before either FA3 protocol.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void probe_counter_permission(int* value) { *value = 1; }

int main() {
    int count = 0;
    int* value = nullptr;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count != 1) {
        std::fprintf(stderr, "Expected exactly one visible GPU; count=%d error=%s\n", count, cudaGetErrorString(status));
        return 1;
    }
    status = cudaMalloc(&value, sizeof(int));
    if (status == cudaSuccess) {
        probe_counter_permission<<<1, 1>>>(value);
        status = cudaDeviceSynchronize();
    }
    if (value) cudaFree(value);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Counter permission kernel failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    return 0;
}
