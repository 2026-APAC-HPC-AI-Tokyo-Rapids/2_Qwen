import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('kernel_tags', ROOT/'scripts/check-kernel-experiment-tags.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('message', ['[cluster:aspire2ap] normal PD', '[skip ci] docs', '[require-build] [cluster:aspire2ap] [build-only]', '[cluster:firmus] [nodes:1] [kernel-bench] candidate'])
def test_explicit_supported_stages(message):
    module.validate(message)


@pytest.mark.parametrize('message', ['[cluster:aspire2ap] [build-only]', '[require-build] [cluster:firmus] [build-only]', '[require-build] [cluster:aspire2ap] [build-only] [profile]', '[cluster:firmus] [kernel-bench]', '[require-build] [cluster:firmus] [nodes:1] [kernel-bench]', '[cluster:firmus] [nodes:1] [kernel-bench] like [dynamo]'])
def test_ambiguous_or_unbounded_stages_are_rejected(message):
    with pytest.raises(ValueError):
        module.validate(message)


@pytest.mark.parametrize("message,pd", [("[cluster:aspire2ap] ordinary", True), ("[require-build] [cluster:aspire2ap] [build-only]", False)])
def test_actual_push_workflow_routes(message, pd):
    import re
    workflow = (ROOT/".github/workflows/qwen.yml").read_text()
    for name, expected in [("aspire2ap", pd)]:
        job_block = re.split(r"\n  [a-zA-Z-]+:\n", workflow.split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]
        expression = re.search(r"^    if: (.+)$", job_block, re.MULTILINE).group(1).removeprefix("${{ ").removesuffix(" }}")
        expression = re.sub(r"contains\(github.event.head_commit.message, '([^']+)'\)", lambda match: str(match[1] in message), expression)
        expression = expression.replace("github.event_name", repr("push")).replace("inputs.run_aspire2ap", "False")
        expression = re.sub(r"needs.container.outputs.\w+", repr("available"), expression)
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        assert eval(expression.strip(), {"__builtins__": {}}, {}) is expected
