#include "verify.h"
#include "cuda_check.h"
namespace apac_fa3_verify {
// B34..40 uses exactly 528 (request,head,split) units. Outside that range
// clamp the same rule; all requests retain at least three exact KV partitions.
__global__ void prepare_uneven(const int* lengths, int* metadata, int b) {
    if (threadIdx.x != 0) { return; }
    int rounded = (b + 3) / 4 * 4;
    int four = max(0, min(b, 132 - 3 * b));
    int order[48], cost[48], count[48];
    // Assign the extra split to the longest requests; deterministic ties by id.
    for (int i = 0; i < b; ++i) { order[i] = i; }
    for (int i = 1; i < b; ++i) {
        int key = order[i], j = i;
        while (j > 0 && lengths[order[j - 1]] < lengths[key]) { order[j] = order[j - 1]; --j; }
        order[j] = key;
    }
    for (int i = 0; i < b; ++i) {
        int request = order[i];
        count[request] = i < four ? 4 : 3;
        cost[request] = ((lengths[request] + 127) / 128 + count[request] - 1) / count[request];
    }
    // Scheduler consumes virtual batch ids sorted by work per split (LPT).
    for (int i = 0; i < b; ++i) { order[i] = i; }
    for (int i = 1; i < b; ++i) {
        int key = order[i], j = i;
        while (j > 0 && cost[order[j - 1]] < cost[key]) { order[j] = order[j - 1]; --j; }
        order[j] = key;
    }
    for (int i = 0; i < b; ++i) {
        int request = order[i], n = max(1, cost[request]);
        metadata[i] = count[request];
        metadata[rounded + i] = 1; // Q4 x GQA16 = M64, exactly one M block.
        metadata[rounded * 2 + i] = request;
        metadata[rounded * 3 + i] = n * 4 <= 256 ? 4 : n * 2 <= 256 ? 2 : 1;
    }
    metadata[rounded * 4] = 0;
}
void decode_prepare(Flash_fwd_params& p, cudaStream_t stream) {
    prepare_uneven<<<1, 32, 0, stream>>>(p.seqused_k, p.num_splits_dynamic_ptr, p.b);
    CHECK_CUDA_KERNEL_LAUNCH();
}
}
