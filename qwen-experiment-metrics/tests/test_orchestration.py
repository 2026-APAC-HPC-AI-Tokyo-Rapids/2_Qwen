from pathlib import Path


def test_workflow_and_scheduler_adapters_enforce_runner_compute_boundary() -> None:
    repository = Path(__file__).parents[2]
    workflow = (repository / ".github/workflows/genkai.yml").read_text()
    aspire = (repository / ".github/workflows/aspire2ap.yml").read_text()
    pjm = (repository / "platforms/genkai/check-qwen-container.pjm").read_text()
    pbs = (repository / "platforms/aspire2ap/check-qwen-container.pbs").read_text()
    check = (repository / "scripts/run-qwen-check-env.sh").read_text()
    provision = (repository / "scripts/provision-qwen-check-env.sh").read_text()

    assert "enroot version" in check
    assert "enroot start" in check
    assert "-m sglang.check_env" in check
    assert "NIXL: nixl=" in check
    assert '--mpi "proc=${{ inputs.node_count }}"' in workflow
    assert "module load intel/2025.1.3 impi" in pjm
    assert 'mpirun_path="$(command -v mpirun)"' in pjm
    assert '"${mpirun_path}" -np "${NODE_COUNT}" -ppn 1' in pjm
    assert 'PMI_RANK:-0' in check
    assert "environment-rank-${rank}.log" in check
    assert "check-env-rank-${rank}.log" in check
    assert "run-success-rank-${rank}" in check
    assert 'for ((rank = 0; rank < NODE_COUNT; rank++))' in pjm
    assert 'printf \'%s\\n\' "${wait_status}"' in workflow
    assert "sleep 15" in workflow
    assert "provision-qwen-check-env.sh" in workflow
    assert "provision-qwen-check-env.sh" in aspire
    assert "configure-aws-credentials" not in workflow
    assert "configure-aws-credentials" not in aspire
    assert "select=${{ inputs.node_count }}:ncpus=8:ngpus=1:mem=128gb:mpiprocs=1" in aspire
    assert "mem=1500gb" not in aspire
    assert "place=scatter:excl" not in aspire
    assert "-l place=scatter " in aspire
    assert "send-qwen-fragments.sh" not in workflow
    assert "send-qwen-fragments.sh" not in aspire
    assert "GOOGLE_CREDENTIAL" not in pjm
    assert "GOOGLE_CREDENTIAL" not in pbs
    assert "AWS_ACCESS_KEY" not in pjm and "AWS_ACCESS_KEY" not in pbs
    assert 'mapfile -t hosts' in pbs
    assert 'EXPECTED_NODE_COUNT=${{ inputs.node_count }}' in aspire
    assert 'node_count="${EXPECTED_NODE_COUNT}"' in pbs
    assert '/opt/pbs/bin/pbsdsh -- "${RUNTIME_DIR}/scripts/run-qwen-check-env-rank.sh"' in pbs
    assert "run-qwen-check-env-rank.sh" in provision
    assert 'rank-map.tsv' in pbs
    assert 'hosts[rank]%%.*' in pbs
    assert 'sort -u | wc -l' in pbs
    assert 'CHECK_ENV_DEFER_AGGREGATE' in check
    assert 'rank_output}/run-success' in check
