# AISimulate SGLang measurement policy adapter

Patch target: [ai-dynamo/aisimulate, commit 992dddc7ee2bad8fd523e7db99eea4ac64254766](https://github.com/ai-dynamo/aisimulate/tree/992dddc7ee2bad8fd523e7db99eea4ac64254766).
Original file: crates/core/src/perfmodel/perf_database/fpm_forward.rs.
Copyright NVIDIA CORPORATION & AFFILIATES, Apache-2.0; see AISimulate-LICENSE.
Modified locally: recognize the explicit `sglang_one_batch_three_repetitions_v1` measurement policy in addition to the native Dynamo policy. Interpolation, workload identity matching and hash validation are unchanged. This is a research adapter owned by this repository; upstream is not modified.

The SGLang producer includes synchronized ModelRunner execution and sampling. Decode medians cover 31 successive forwards and are assigned the center KV coordinate (ISL+16). These assumptions are retained in the sidecar and must be checked against held-out model shapes and independent serving runs.

Build in a disposable checkout of the pinned revision with uv:

```sh
git apply /path/to/2_Qwen/research/simulation/patches/aisimulate-sglang-measurement-policy.patch
uv build --wheel --out-dir /tmp/aisimulate-wheels python/aisimulate
uv pip install --python /path/to/simulation-venv/bin/python --no-deps /tmp/aisimulate-wheels/aisimulate-0.13.0-cp311-abi3-linux_x86_64.whl
```

This CPU simulator rebuild does not change the GPU serving container.
