import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('kernel_tags', ROOT/'scripts/check-kernel-experiment-tags.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('message', ['[cluster:aspire2ap] normal PD', '[skip ci] docs', '[require-build] [cluster:aspire2ap] [build-only]', '[cluster:aspire2ap] [nodes:1] [cutlass-grid-probe]', '[cluster:aspire2ap] [nodes:1] [moe-routing-probe]'])
def test_explicit_supported_stages(message):
    module.validate(message)


@pytest.mark.parametrize('message', ['[require-build] [cluster:aspire2ap] [build-only] [cutlass-grid-probe]', '[cluster:aspire2ap] [build-only]', '[cluster:aspire2ap] [cutlass-grid-probe]', '[cluster:aspire2ap] [nodes:1] [cutlass-grid-probe] [profile]', '[require-build] [cluster:aspire2ap] [nodes:1] [cutlass-grid-probe]', '[require-build] [cluster:firmus] [build-only]', '[cluster:aspire2ap] [nodes:1] [nodes:2] [cutlass-grid-probe]', '[cluster:aspire2ap] [nodes:1] [cutlass-grid-probe] [moe-routing-probe]', '[require-build] [cluster:aspire2ap] [build-only] [moe-routing-probe]'])
def test_ambiguous_or_unbounded_stages_are_rejected(message):
    with pytest.raises(ValueError):
        module.validate(message)


def test_gpu_only_path_has_no_official_score_or_unallocated_devices():
    driver=(ROOT/'platforms/aspire2ap/run-cutlass-grid.pbs').read_text()
    assert 'walltime=00:15:00' in driver and 'timeout 720' in driver
    assert 'NVIDIA_VISIBLE_DEVICES=${uuid}' in driver and 'NVIDIA_VISIBLE_DEVICES=all' not in driver
    assert 'CUDA_VISIBLE_DEVICES=0' in driver
    workflow=(ROOT/'.github/workflows/aspire2ap-grid.yml').read_text()
    assert 'select=1:ncpus=14:ngpus=1:mem=235gb:mpiprocs=1' in workflow
    assert 'place=free' in workflow and 'host=a2ap-' not in workflow
    assert 'check-qwen-job.sh' not in workflow and 'profile_export_aiperf' not in workflow
    check=(ROOT/'platforms/aspire2ap/check-cutlass-grid.sh').read_text()
    assert 'test ! -e "${RUNTIME_DIR}/results/score.json"' in check


@pytest.mark.parametrize("message,pd,probe", [("[cluster:aspire2ap] ordinary", True, False), ("[require-build] [cluster:aspire2ap] [build-only]", False, False), ("[cluster:aspire2ap] [nodes:1] [cutlass-grid-probe]", False, True), ("[cluster:aspire2ap] [nodes:1] [moe-routing-probe]", False, True)])
def test_actual_push_workflow_routes(message, pd, probe):
    import re
    workflow = (ROOT/".github/workflows/qwen.yml").read_text()
    for name, expected in [("aspire2ap", pd), ("cutlass-grid", probe)]:
        job_block = re.split(r"\n  [a-zA-Z-]+:\n", workflow.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]
        expression = re.search(r"^    if: (.+)$", job_block, re.MULTILINE).group(1).removeprefix("${{ ").removesuffix(" }}")
        expression = re.sub(r"contains\(github.event.head_commit.message, '([^']+)'\)", lambda match: str(match[1] in message), expression)
        expression = expression.replace("github.event_name", repr("push")).replace("inputs.run_aspire2ap", "False")
        expression = re.sub(r"needs.container.outputs.\w+", repr("available"), expression)
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        assert eval(expression.strip(), {"__builtins__": {}}, {}) is expected
