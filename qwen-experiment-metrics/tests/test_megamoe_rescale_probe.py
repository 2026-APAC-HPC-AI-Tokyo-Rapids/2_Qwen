"""CPU contracts for the isolated diagnostic, with no git-history dependency."""
import gzip
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/probe-firmus-megamoe-rescale.py'
spec = importlib.util.spec_from_file_location('rescale_probe', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def host(tmp_path):
    files = json.loads(gzip.decompress((ROOT / 'qwen-experiment-metrics/tests/fixtures/megamoe-rescale-r41-host.json.gz').read_bytes()))
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(['git', 'apply', '--check', str(ROOT / 'patches/deep-gemm-host/sm90-rescale-exposure-probe-r41.patch')], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(['git', 'apply', str(ROOT / 'patches/deep-gemm-host/sm90-rescale-exposure-probe-r41.patch')], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path, files


def generator(text):
    start = text.index('    static std::string generate_impl(')
    return text[start:text.index('    static void launch_impl(', start)]


def test_production_generator_is_identical_and_diagnostic_is_only_prefix(host):
    directory, files = host
    name = 'csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp'
    patched = (directory / name).read_text()
    prefix = r'''
#include <cassert>
#include <sstream>
#include <string>
namespace fmt { template<class... T> std::string format(const char* s, T... args) { std::ostringstream o; o<<s; ((o<<args<<";"),...); return o.str(); } }
std::string to_string(float x) { return std::to_string(x); }
struct Config { int num_experts_per_wave=128,block_m=128,block_n=256,block_k=128,num_max_pool_tokens=92160,num_padded_sf_pool_tokens=1474560,num_stages=4,num_dispatch_threads=64,num_non_epilogue_threads=64,num_epilogue_threads=256; };
struct Args {
 int num_max_tokens_per_rank=8448,hidden=4096,intermediate_hidden=1536,num_experts=128,num_topk=8,num_ranks=1,epilogue_registers=0;
 float activation_clamp=0; bool fast_math=true,reuse_accum_as_final=true,l2_arrival_counter=true,l2_epilogue_requires_full_sync=false,split_phase_hot_path=false,use_swap_ab=false,prefill_lean=true,prefill_tail_pair=false,prefill_split_accum=true,prefill_fc1_first_k=true,diagnostic_no_rescale=false;
 Config config; struct { std::pair<int,int> grid_dim={132,1}; } launch_args;
};
'''
    code = prefix + 'struct Before {\n' + generator(files[name]) + '};\nstruct After {\n' + generator(patched) + '};\n' + r'''
int main() {
 for (bool first: {false,true}) for (bool split: {false,true}) for (bool lean: {false,true}) for (bool tail: {false,true}) {
  Args a; a.prefill_fc1_first_k=first; a.prefill_split_accum=split; a.prefill_lean=lean; a.prefill_tail_pair=tail;
  auto production=Before::generate_impl(a); assert(After::generate_impl(a)==production);
  a.diagnostic_no_rescale=true;
  assert(After::generate_impl(a)=="#define DG_R41_DIAGNOSTIC_NO_RESCALE 1\n#define sm90_fp8_mega_moe_impl sm90_fp8_mega_moe_rescale_exposure_diagnostic_impl\n"+production);
 }
}
'''
    cpp = directory / 'generator.cpp'
    cpp.write_text(code)
    subprocess.run(['g++', '-std=c++17', '-O2', str(cpp), '-o', str(directory / 'generator')], check=True, capture_output=True)
    subprocess.run([str(directory / 'generator')], check=True)


def test_host_production_has_no_diagnostic_environment_route(host):
    directory, files = host
    ffi = (directory / 'csrc/tvm_ffi_api.cpp').read_text()
    assert 'TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mega_moe, dg_fp8_mega_moe<false>);' in ffi
    assert 'TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mega_moe_rescale_exposure_diagnostic, dg_fp8_mega_moe<true>);' in ffi
    api = (directory / 'csrc/apis/sm90_mega.hpp').read_text()
    runtime = (directory / 'csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp').read_text()
    assert 'template <bool DiagnosticNoRescale = false>' in api and 'template <bool DiagnosticNoRescale = false>' in runtime
    assert 'sm90_fp8_mega_moe<DiagnosticNoRescale>(y,' in api
    assert '.diagnostic_no_rescale = DiagnosticNoRescale,' in runtime
    assert 'DG_HOST_ASSERT(prefill_fc1_first_k and num_tokens == 7988 and num_ranks == 1 and num_topk == 8)' in runtime
    assert f'get_env<int>("{probe.FLAG}"' not in runtime
    assert 'fp8_mega_moe_rescale_exposure_diagnostic' not in files['sgl_deep_gemm/__init__.py']
    config = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    # No SGLang overlay is allowed to reference the diagnostic API or flag.
    for patch in config['sglang_overlay_patches']:
        text = (ROOT / patch).read_text()
        assert probe.FLAG not in text and probe.SYMBOL not in text
    assert 'sm90-rescale-exposure-probe-r41.patch' in (ROOT / 'patches/deep-gemm-host/series').read_text()


def test_flag_rejected_before_any_gpu_import(monkeypatch):
    for key, value in probe.REQUIRED.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(probe.FLAG, '0')
    with pytest.raises(AssertionError, match='never be in the serving'):
        probe.run_worker(None, {})


def source_pair():
    production = "// Includes' hash value: same\n#define DG_SM90_PREFILL_FC1_FIRST_K 1\nstatic void __instantiate_kernel() { auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_impl<" + probe.TEMPLATE + '>); };\n'
    diagnostic = production.replace('#define DG_SM90_PREFILL_FC1_FIRST_K', probe.PREFIX + '#define DG_SM90_PREFILL_FC1_FIRST_K')
    return production, diagnostic


def test_real_cache_source_assertions(tmp_path, monkeypatch):
    production, diagnostic = source_pair()
    probe.assert_jit_sources(production, diagnostic)
    for bad in (production.replace('8448', '8192'), production.replace('FIRST_K 1', 'FIRST_K 0'), production.replace('hash value: same', 'hash value: other')):
        with pytest.raises(AssertionError):
            probe.assert_jit_sources(bad, diagnostic)
    for variant, source in (('production', production), ('diagnostic', diagnostic)):
        p = tmp_path / f'cache/kernel.sm90_fp8_mega_moe.{variant}'
        p.mkdir(parents=True)
        (p / 'kernel.cu').write_text(source)
    def symbols(cmd, **kwargs):
        name = probe.SYMBOL if 'diagnostic' in str(cmd[-1]) else 'sm90_fp8_mega_moe_impl'
        return 'STT_FUNC STO_ENTRY ' + name + '\n'
    monkeypatch.setattr(subprocess, 'check_output', symbols)
    identities = probe.cache_identity(tmp_path, Path('/cuda/bin/cuobjdump'))
    assert set(identities) == {'production', 'diagnostic'}
    assert all(r['jit_cache_key'] and r['kernel_name'] for r in identities.values())


def samples(delta):
    return [{'order': order, 'round': r, 'production_us': 2600 + r, 'diagnostic_us': 2600 + r - delta} for order in ('forward', 'reverse') for r in range(12)]


def populate(directory, delta=49.9):
    for worker in range(3):
        for rank in range(4):
            probe.write(directory / f'worker-{worker}/rank-{rank}.json', {'schema_version': probe.SCHEMA, 'status': 'complete', 'worker': worker, 'rank': rank, 'run_id': 'run', 'image_sha256': 'image', 'routing_sha256': probe.FIXTURE_SHA, 'gpu_uuid': f'GPU-{worker}-{rank}', 'serving_environment': probe.REQUIRED, 'jit': {'production': {'jit_cache_key':'prod','kernel_name':'sm90_fp8_mega_moe_impl'}, 'diagnostic':{'jit_cache_key':'diag','kernel_name':probe.SYMBOL}}, 'sm_clock_samples_mhz': [1400, 1420], 'samples': samples(delta)})


@pytest.mark.parametrize('delta,decision', [(49.9, 'closed'), (50, 'conditionally_open'), (100, 'conditionally_open'), (-2, 'closed')])
def test_twelve_gpu_schema_and_decision(tmp_path, delta, decision):
    populate(tmp_path, delta)
    result = probe.aggregate(tmp_path, 'run', 'image', [0, 0, 0])
    assert result['status'] == 'complete' and result['decision'] == decision
    assert result['median_12gpu_delta_us'] == pytest.approx(delta)
    assert len(result['gpus']) == 12
    assert result['gpus'][0]['clocks_per_steady_k128_cta_step'] == pytest.approx(delta * 132 * 1410 / 292358)
    assert 'upper bound' in result['interpretation'] and 'not separated' in result['remaining_uncertainty']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['missing', 'clock', 'run', 'image', 'uuid', 'rounds', 'jit', 'flag', 'nan', 'timeout'])
def test_incomplete_is_not_a_zero_exposure(tmp_path, failure):
    populate(tmp_path)
    path = tmp_path / 'worker-2/rank-3.json'
    row = json.loads(path.read_text())
    if failure == 'missing':
        path.unlink()
    else:
        if failure == 'clock': row['sm_clock_samples_mhz'] = []
        if failure == 'run': row['run_id'] = 'other'
        if failure == 'image': row['image_sha256'] = 'other'
        if failure == 'uuid': row['gpu_uuid'] = 'GPU-0-0'
        if failure == 'rounds': row['samples'].pop()
        if failure == 'jit': row['jit'] = {}
        if failure == 'flag': row['serving_environment'][probe.FLAG] = '1'
        if failure == 'nan': row['samples'][0]['production_us'] = float('nan')
        path.write_text(json.dumps(row))
    result = probe.aggregate(tmp_path, 'run', 'image', [0, 124, 0] if failure == 'timeout' else [0, 0, 0])
    assert result['status'] == 'incomplete' and 'median_12gpu_delta_us' not in result


def test_sbatch_budget_staging_and_artifact_collection():
    batch = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    start = batch.index('# Diagnostic only: the freed r37 window')
    end = batch.index('# FC1 first-K is pinned', start)
    block = batch[start:end]
    assert batch.index('# Decode greedy gate end.') < start < batch.index('bash "${ROOT}/source/worker.sh" prefill-0')
    assert 'SECONDS + 105 + 630 <= 1020' in block
    assert 'timeout --signal=KILL 105 bash' in block
    assert '"status":"skipped"' in block and 'status:"incomplete"' in block
    assert 'prefill-kernels.env' not in block and 'env-output' not in block
    launch = (ROOT / 'scripts/launch-firmus-megamoe-rescale-probe.sh').read_text()
    assert 'timeout --foreground --kill-after=2 95 srun' in launch
    assert 'for worker in 0 1 2; do' in launch and 'pids+=("$!")' in launch
    assert 'if (( worker == 2 )); then node=${tail_node}; fi' in launch
    wrapper = (ROOT / 'scripts/run-firmus-megamoe-rescale-probe.sh').read_text()
    assert '--nproc-per-node=4' in wrapper and 'source "${root}/source/prefill-kernels.env"' in wrapper
    assert probe.FLAG not in wrapper
    provision = (ROOT / 'scripts/provision-firmus-framework.sh').read_text()
    for name in ('probe-firmus-megamoe-rescale.py', 'launch-firmus-megamoe-rescale-probe.sh', 'run-firmus-megamoe-rescale-probe.sh', 'megamoe-local-ep1-routing.npz'):
        assert name in provision
    workflow = (ROOT / '.github/workflows/firmus.yml').read_text()
    assert 'for part in logs profile results source; do' in workflow
    assert '${RUNTIME_DIR}/${part}' in workflow
    # Each timing capture calls the unchanged production API, with JIT/prewarm off-clock.
    source = SCRIPT.read_text()
    assert source.index('jit = cache_identity(') < source.index('start.record()')
    assert 'production = deep_gemm.fp8_mega_moe' in source
    assert 'with torch.cuda.graph(graph):' in source
    assert 'start.elapsed_time(end) * 1000 / CALLS' in source


@pytest.mark.parametrize('fail_worker', [False, True])
def test_cpu_supervisor_launches_three_workers_concurrently_and_aggregates(tmp_path, fail_worker):
    import os
    import shutil
    import sys

    runtime = tmp_path / 'runtime'
    (runtime / 'scripts').mkdir(parents=True)
    (runtime / 'logs').mkdir()
    shutil.copy2(SCRIPT, runtime / 'scripts' / SCRIPT.name)
    populate(runtime / 'results/megamoe-rescale-probe')
    for worker in range(3):
        for path in (runtime / f'results/megamoe-rescale-probe/worker-{worker}').glob('rank-*.json'):
            row = json.loads(path.read_text())
            row['run_id'], row['image_sha256'] = 'run', 'image'
            path.write_text(json.dumps(row))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    scripts = {
        'timeout': 'import os,sys\nos.execvp(sys.argv[4],sys.argv[4:])\n',
        'srun': '''import os,sys,time
from pathlib import Path
root,worker,run,image=sys.argv[-4:]
root=Path(root)
(root/('started-'+worker)).touch()
deadline=time.monotonic()+2
while len(list(root.glob('started-*'))) != 3:
 if time.monotonic()>deadline: sys.exit(19)
 time.sleep(.01)
sys.exit(42 if os.environ.get('MOCK_FAIL')=='1' and worker=='1' else 0)
''',
        'uv': 'import os,sys\nindex=next(i for i,a in enumerate(sys.argv) if a.endswith(".py"))\nos.execv(sys.executable,[sys.executable,*sys.argv[index:]])\n',
    }
    for name, source in scripts.items():
        p = bin_dir / name
        p.write_text('#!' + sys.executable + '\n' + source)
        p.chmod(0o755)
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'], 'MOCK_FAIL': str(int(fail_worker))}
    subprocess.run(['bash', str(ROOT / 'scripts/launch-firmus-megamoe-rescale-probe.sh'), str(runtime), 'head', 'tail', 'image.sqsh', 'run', 'image', 'm0', 'm1', 'm2'], check=True, env=env, timeout=10, capture_output=True)
    summary = json.loads((runtime / 'results/megamoe-rescale-probe/summary.json').read_text())
    assert summary['worker_exit_codes'] == ([0, 42, 0] if fail_worker else [0, 0, 0])
    assert summary['status'] == ('incomplete' if fail_worker else 'complete')
    assert len(list(runtime.glob('started-*'))) == 3
    assert not list(runtime.rglob('*.env'))


@pytest.mark.parametrize('elapsed,launch', [(285, True), (286, False)])
def test_sbatch_budget_boundary_runs_or_writes_skip(tmp_path, elapsed, launch):
    import os
    import shutil
    import sys

    if not shutil.which('jq'):
        pytest.skip('jq required by sbatch')
    batch = (ROOT / 'platforms/firmus/framework-comparison/sglang.sbatch').read_text()
    block = batch[batch.index('# Diagnostic only: the freed r37 window'):batch.index('# FC1 first-K is pinned')]
    (tmp_path / 'source').mkdir()
    (tmp_path / 'source/firmus-framework.json').write_text('{"images":{"sglang":{"sha256":"image"}}}')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    timeout = bin_dir / 'timeout'
    # Mock only: no scheduler or GPU command can execute in this fixture.
    timeout.write_text('#!' + sys.executable + '\nimport os,sys\nfrom pathlib import Path\nPath(os.environ["ROOT"],"launched").touch()\nsys.exit(137)\n')
    timeout.chmod(0o755)
    env = {**os.environ, 'ROOT': str(tmp_path), 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH']}
    command = 'prefill_pp=4\nHEAD=head\nTAIL=tail\nSGLANG=image\nSLURM_JOB_ID=job\nmounts() { echo mount; }\nSECONDS=' + str(elapsed) + '\n' + block
    subprocess.run(['bash', '-euc', command], check=True, env=env, capture_output=True)
    summary = json.loads((tmp_path / 'results/megamoe-rescale-probe/summary.json').read_text())
    assert (tmp_path / 'launched').exists() == launch
    assert summary['status'] == ('incomplete' if launch else 'skipped')
    if launch:
        assert summary['exit_code'] == 137


def test_runtime_overlay_changes_only_the_accepted_rescale_guards(host):
    directory, files = host
    name = 'deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh'
    subprocess.run(['git', 'apply', '--check', str(ROOT / 'patches/deep-gemm-overlay/sm90-rescale-exposure-probe-r41.patch')], cwd=directory, check=True, capture_output=True)
    subprocess.run(['git', 'apply', str(ROOT / 'patches/deep-gemm-overlay/sm90-rescale-exposure-probe-r41.patch')], cwd=directory, check=True, capture_output=True)
    source = (directory / name).read_text()
    assert source.count(f'if constexpr (!{probe.FLAG}) final_accum[') == 8
    restored = source.replace(f'#ifndef {probe.FLAG}\n#define {probe.FLAG} 0\n#endif\n\n', '').replace(f'if constexpr (!{probe.FLAG}) ', '')
    assert restored == files[name]
    config = json.loads((ROOT / 'configs/firmus-prefill-pp4.json').read_text())
    assert config['deep_gemm_overlay_patches'][-1] == 'patches/deep-gemm-overlay/sm90-rescale-exposure-probe-r41.patch'


def test_final_overlay_cpu_compile_proof_matches_the_accepted_binary():
    import hashlib
    import tarfile

    proof = ROOT / 'research/runtime-performance/megamoe-rescale-probe-r41-cpu-proof-20261104.tar.gz'
    with tarfile.open(proof) as archive:
        for manifest, prefix in (('r41-proof/hashes.json', 'r41-proof/'), ('r41b-proof/hashes.json', '')):
            for name, digest in json.load(archive.extractfile(manifest)).items():
                assert hashlib.sha256(archive.extractfile(prefix + name).read()).hexdigest() == digest
        compiled = json.load(archive.extractfile('r41b-proof/compile-results.json'))
        accepted = json.load(archive.extractfile('r41-proof/compile-results.json'))
        for name in ('production', 'diagnostic'):
            assert compiled[name]['registers'] == ['168']
            assert compiled[name]['spills'] == [['0', '0']]
            assert compiled[name]['stack'] == ['0']
            assert compiled[name]['matches_accepted_r41_proof'] is True
            assert compiled[name]['encoding_sha256'] == accepted[name]['encoding_sha256']
        delta = {op: accepted['diagnostic']['opcodes'].get(op, 0) - accepted['production']['opcodes'].get(op, 0) for op in ('FMUL', 'MUFU.RCP', 'LDG.E.CONSTANT', 'LDS', 'QGMMA.64', 'UTMALDG.2D', 'WARPGROUP.ARRIVE', 'WARPGROUP.DEPBAR.LE')}
        assert delta == {'FMUL': -3552, 'MUFU.RCP': -108, 'LDG.E.CONSTANT': -52, 'LDS': -52, 'QGMMA.64': 0, 'UTMALDG.2D': 0, 'WARPGROUP.ARRIVE': 0, 'WARPGROUP.DEPBAR.LE': 0}
