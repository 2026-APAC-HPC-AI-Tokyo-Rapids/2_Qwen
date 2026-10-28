import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('kernel_tags', ROOT/'scripts/check-kernel-experiment-tags.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('message', ['[cluster:aspire2ap] normal PD', '[skip ci] docs', '[require-build] [cluster:aspire2ap] [build-only]', '[require-build] [cluster:firmus] [build-only]', '[cluster:firmus] [nodes:1] [kernel-bench] candidate'])
def test_explicit_supported_stages(message):
    module.validate(message)


@pytest.mark.parametrize('message', ['[cluster:aspire2ap] [build-only]', '[cluster:firmus] [build-only]', '[require-build] [build-only]', '[require-build] [cluster:genkai] [build-only]', '[require-build] [cluster:firmus] [cluster:aspire2ap] [build-only]', '[require-build] [cluster:firmus] [build-only] [dynamo]', '[require-build] [cluster:firmus] [build-only] [profile]', '[require-build] [cluster:aspire2ap] [build-only] [profile]', '[cluster:firmus] [kernel-bench]', '[require-build] [cluster:firmus] [nodes:1] [kernel-bench]', '[require-build] [cluster:firmus] [nodes:1] [kernel-bench] [build-only]', '[cluster:firmus] [nodes:1] [kernel-bench] like [dynamo]'])
def test_ambiguous_or_unbounded_stages_are_rejected(message):
    with pytest.raises(ValueError):
        module.validate(message)


@pytest.mark.parametrize("cluster", ["aspire2ap", "firmus"])
@pytest.mark.parametrize("stage", ["normal", "build-only", "kernel-bench"])
def test_actual_push_workflow_routes(cluster, stage):
    import re
    message = (f"[cluster:{cluster}] [nodes:1] [kernel-bench]" if stage == "kernel-bench" else f"[require-build] [cluster:{cluster}]" + (" [build-only]" if stage == "build-only" else ""))
    module.validate(message)
    workflow = (ROOT/".github/workflows/qwen.yml").read_text()
    for name in ["container", "genkai", "aspire2ap", "firmus", "firmus-dynamo", "firmus-kernel-bench", "aspire2ap-kernel-bench"]:
        expected = name == f"{cluster}-kernel-bench" if stage == "kernel-bench" else name == "container" or (name == cluster and stage == "normal")
        job_block = re.split(r"\n  [a-zA-Z0-9-]+:\n", workflow.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]
        expression = re.search(r"^    if: (.+)$", job_block, re.MULTILINE).group(1).removeprefix("${{ ").removesuffix(" }}")
        expression = re.sub(r"contains\(github.event.head_commit.message, '([^']+)'\)", lambda match: str(match[1] in message), expression)
        expression = expression.replace("github.event_name", repr("push"))
        expression = re.sub(r"inputs.run_\w+", "False", expression)
        expression = re.sub(r"needs.container.outputs.\w+", repr("available"), expression)
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        assert eval(expression.strip(), {"__builtins__": {}}, {}) is expected


@pytest.mark.parametrize('cluster', ['aspire2ap', 'firmus'])
def test_kernel_bench_supported_cluster(cluster):
    module.validate(f'[cluster:{cluster}] [nodes:1] [kernel-bench]')


@pytest.mark.parametrize('suffix', ['[nodes:2]', '[nodes:4]', '[profile]', '[cluster:genkai]', '[require-build]'])
def test_aspire_kernel_bench_rejects_unexecuted_options(suffix):
    with pytest.raises(ValueError):
        module.validate(f'[cluster:aspire2ap] [nodes:1] [kernel-bench] {suffix}')
