"""CPU model of the direct-register epilogue; no GPU dependency."""

import numpy as np

OUTPUTS = ("q", "k", "v", "k_cache", "v_cache")


def fp8_codes(x):
    """Independent round-to-nearest-even E4M3FN, CUDA NOSAT semantics."""
    x = np.asarray(x)
    code = np.arange(127, dtype=np.int32)
    exp, frac = code >> 3, code & 7
    values = np.where(exp == 0, frac * 2.0**-9, (1 + frac / 8) * np.exp2(exp - 7))
    mag = np.abs(x).astype(np.float64)
    upper = np.minimum(np.searchsorted(values, mag), 126)
    lower = np.maximum(upper - 1, 0)
    du, dl = np.abs(values[upper] - mag), np.abs(mag - values[lower])
    nearest = np.where((du < dl) | ((du == dl) & ((upper & 1) == 0)), upper, lower)
    sign = np.signbit(x).astype(np.uint8) << 7
    out = nearest.astype(np.uint8) | sign
    out = np.where(mag > 464, np.uint8(127) | sign, out)
    return np.where(np.isnan(x), np.uint8(127), out).astype(np.uint8)


def bf16_round(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def epilogue(acc, weight, trig, *, fp64=False, bf16_input=False, eps=1e-6):
    """One Q/K head tile. Trig columns are cos[0:64], sin[0:64]."""
    dtype = np.float64 if fp64 else np.float32
    acc = np.asarray(acc, dtype=dtype)
    if bf16_input:
        acc = bf16_round(acc).astype(dtype)
    weight = np.asarray(weight, dtype=dtype)
    trig = np.asarray(trig, dtype=dtype)
    if fp64:
        sums = (acc * acc).sum(axis=-1)
    else:
        # WGMMA quad ownership: pairs 2*lane+8*j, plus 64.
        partial = np.zeros((*acc.shape[:-1], 4), dtype=dtype)
        for lane in range(4):
            chains = []
            for half in (0, 64):
                value = np.zeros(acc.shape[:-1], dtype=dtype)
                for j in range(8):
                    for e in range(2):
                        x = acc[..., half + 8 * j + 2 * lane + e].astype(np.float64)
                        value = (x * x + value.astype(np.float64)).astype(dtype)
                chains.append(value)
            partial[..., lane] = chains[0] + chains[1]
        partial = partial + partial[..., [2, 3, 0, 1]]
        partial = partial + partial[..., [1, 0, 3, 2]]
        sums = partial[..., 0]
    rho = (1 / np.sqrt(sums * dtype(1 / 128) + dtype(eps))).astype(dtype)
    u, v = acc[..., :64] * weight[:64], acc[..., 64:] * weight[64:]
    c, s = trig[..., :64], trig[..., 64:]
    result = np.concatenate(((u * c - v * s) * rho[..., None], (u * s + v * c) * rho[..., None]), axis=-1)
    return result, fp8_codes(result)


def ownership():
    """(thread, wave, row-in-fragment, column-chunk, element) -> row,column,index."""
    for t in range(256):
        for wave in range(2):
            for r in range(2):
                for j in range(16):
                    for e in range(2):
                        yield t, wave, r, 16 * (t // 32) + (t % 32) // 4 + wave * 128 + r * 8, 8 * j + 2 * (t % 4) + e, wave * 64 + 4 * j + 2 * r + e
