#pragma once
#include <cmath>
#ifdef __CUDACC__
#define GREEDY_HD __host__ __device__
#else
#define GREEDY_HD
using std::isnan;
using std::isinf;
#endif
struct GreedyPair { float value; int index; };
GREEDY_HD inline GreedyPair greedy_better(GreedyPair a, GreedyPair b) {
    // Torch argmax: first NaN wins, then greatest value, then smallest index.
    const bool an=isnan(a.value), bn=isnan(b.value);
    if (an != bn) return an ? a : b;
    if (an || a.value == b.value) return a.index < b.index ? a : b;
    return a.value > b.value ? a : b;
}
GREEDY_HD inline float greedy_value(float x, bool sanitize, bool penalty, bool stop) {
    if (sanitize) {
        if (isnan(x)) x=-1e30f;
        else if (isinf(x)) x=x>0 ? 1e30f : -1e30f;
    }
    if (penalty) {
#ifdef __CUDA_ARCH__
        x=__fadd_rn(x,stop ? -INFINITY : 0.f);
#else
        x=x+(stop ? -INFINITY : 0.f);
#endif
    }
    return x;
}

#undef GREEDY_HD
