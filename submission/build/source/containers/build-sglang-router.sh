#!/usr/bin/env bash
# Build the pinned native router and test its Prefill accounting on the build CPU.
set -euxo pipefail

work=/var/tmp/sglang-router-build
install -d "${work}" /opt/apac/bin
trap 'rm -rf -- "${work}"' EXIT
curl -fsSL https://files.pythonhosted.org/packages/e2/14/19020d822877810d1b047073bb41c54a76802e618296af12344f6caa6d2e/sglang_router-0.3.2.tar.gz -o "${work}/router.tar.gz"
printf 'bdbea1d54cce879fb83d2885d01ee3c006f242d0adbbac806f2fb2b91a5fe73d  %s\n' "${work}/router.tar.gz" | sha256sum --check
tar -xzf "${work}/router.tar.gz" -C "${work}"
cd "${work}/sglang_router-0.3.2"
patch -p1 < /opt/apac/prefill-load-lifetime.patch
patch -p1 < /opt/apac/no-delay-acceptor.patch
cp /opt/apac/sglang-router.Cargo.lock Cargo.lock
export CARGO_HOME="${work}/cargo" RUSTUP_HOME="${work}/rustup"
curl -fsSL https://sh.rustup.rs -o "${work}/rustup-init.sh"
sh "${work}/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain 1.95.0
export PATH="${CARGO_HOME}/bin:${PATH}"
cargo test --locked --profile ci --lib routers::http::pd_router::tests
cargo test --locked --profile ci --lib policies::manual::tests
cargo build --locked --profile ci --bin smg
install -m 0755 target/ci/smg /opt/apac/bin/sglang-router
/opt/apac/bin/sglang-router launch --help >/dev/null
python3 -c 'import hashlib,json,pathlib; names={"patch_sha256":"prefill-load-lifetime.patch","nodelay_patch_sha256":"no-delay-acceptor.patch","cargo_lock_sha256":"sglang-router.Cargo.lock","binary_sha256":"bin/sglang-router"}; result={"version":"0.3.2","source_sha256":"bdbea1d54cce879fb83d2885d01ee3c006f242d0adbbac806f2fb2b91a5fe73d","rust_version":"1.95.0","prefill_load":"dispatch_through_body_completion"}; result.update({key:hashlib.sha256((pathlib.Path("/opt/apac")/name).read_bytes()).hexdigest() for key,name in names.items()}); pathlib.Path("/opt/apac/sglang-router-build.json").write_text(json.dumps(result,indent=2)+"\n")'
