# vLLM image build on AWS

After merging the workflow into `main`, run **AWSでvLLMイメージをビルド** (`vllm-image.yml`) from GitHub Actions using **Run workflow**.
GitHub requires a new manual workflow to exist on the default branch before it can be dispatched; a feature-branch push only runs the existing repository validation workflow.
The workflow is manual and builds the version tracked by the selected commit.
It validates the repository, stages the checkout on S3, invokes the shared AWS builder, and distributes the verified SquashFS to Firmus.
It does not launch an inference or benchmark job.

The existing repository variables are reused: `QWEN_AWS_BUILD_ROLE_ARN`, `QWEN_AWS_BUILD_INSTANCE_PROFILE`, `QWEN_AWS_BUILD_SECURITY_GROUP`, `QWEN_AWS_BUILD_CACHE_REPOSITORY`, and `QWEN_CONTAINER_BUCKET`.
No additional AWS resources or GitHub credentials are needed for this path.

## Image definition

[`containers/vllm.Dockerfile`](../containers/vllm.Dockerfile) derives from the official `vllm/vllm-openai:v0.30.0` image used by the successful Firmus baseline.
Its base digest is `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`.
The recipe preserves the official serving entrypoint and installed vLLM, Torch, and NIXL packages.
It selects `/usr/local/cuda/bin/nvcc` to match CUDA 13.0 runtime headers and writes PATH and GPU injection variables to `/etc/environment` for Enroot.
The build does not compile vLLM from source or bundle model weights.
Version changes belong in the recipe and its CPU validation script, so the Git commit identifies each build condition.

[`scripts/build-vllm-container.sh`](../scripts/build-vllm-container.sh) runs rootful Buildah with chroot isolation and OverlayFS, using the existing ECR layer cache.
The CPU check verifies imports, pinned package versions, the CUDA compiler, the Enroot environment, and the embedded build manifest.
Skopeo and Umoci validate the OCI layout before Docker archive export and SquashFS creation.
GPU serving and NIXL transfers require separate Slurm validation on Firmus; the CPU-only EC2 builder cannot certify them.

## Artifacts

The shared builder records S3 locations in the Actions job summary:

- `containers/<commit>/<run-id>-<attempt>/vllm.docker.tar.zst`: a Docker-loadable archive compressed with Zstandard.
- `containers/<commit>/<run-id>-<attempt>/vllm.sqsh`: the Enroot/Pyxis runtime filesystem.
- `containers/<commit>/<run-id>-<attempt>/container-sidecar.json`: sizes, SHA-256 checksums, Docker tag, and provenance.

These objects inherit the existing S3 one-day lifecycle.
Download the Docker archive before it expires, verify its SHA-256 against the sidecar, and load it with:

```sh
zstd -dc vllm.docker.tar.zst | docker load
```

The loaded image is tagged `apac-qwen-vllm:<commit>`.
The Docker archive is saved on S3; the ECR repository continues to hold Buildah cache layers rather than a final serving tag.

Firmus receives `<run-id>.sqsh` and `<run-id>.container-sidecar.json` under `/scratch/home/ACCOUNT/qwen/vllm-containers`.
This archive is separate from the SGLang archive and the existing `framework-comparison/images` baselines.
The workflow verifies both size and SHA-256 before publishing the files.
The existing Firmus comparison workflow still selects its pinned baseline from `configs/firmus-framework.json`; it does not switch images after a build.
To test a new image, record its path, size, SHA-256, and sidecar path in a separate experiment commit before launching the existing comparison workflow.
The comparison workflow's `[require-build]` guard remains in place; this image build is a separate workflow.

EC2 bootstrapping, SSM execution, termination, build-log collection, and cost estimation are shared through [`aws-image-build.yml`](../.github/workflows/aws-image-build.yml).
The SGLang workflow calls the same builder with its default framework.
