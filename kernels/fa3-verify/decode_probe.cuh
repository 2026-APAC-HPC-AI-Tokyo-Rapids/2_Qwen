#pragma once
// Defined only by the standalone Decode merge probe, never by build.py.
#ifdef FA3_DECODE_MERGE_PROBE
struct DecodeProbe {
    int* records = nullptr; // [batch, kv_head, split, 12]
    int64_t* abi = nullptr; // [batch, kv_head, split, 28]
    float* loaded = nullptr;
    float* loaded_lse = nullptr; // [2, split, head, token]: max and weight loads
    float* merged = nullptr; // [token, head, dim]
    float* amax = nullptr; // [head, token]
    uint8_t* codes = nullptr;
    float* scales = nullptr;
};
#endif
