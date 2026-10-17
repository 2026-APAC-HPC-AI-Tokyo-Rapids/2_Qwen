import json
from pathlib import Path


def test_workflows_keep_the_shared_pd_result_and_archive_contract() -> None:
    repository = Path(__file__).parents[2]
    genkai = (repository / ".github/workflows/genkai.yml").read_text()
    aspire = (repository / ".github/workflows/aspire2ap.yml").read_text()
    firmus = (repository / ".github/workflows/firmus.yml").read_text()
    submission = (repository / "submission/run/submit.sbatch").read_text()
    genkai_job = (repository / "platforms/genkai/run-qwen-pd.pjm").read_text()
    aspire_job = (repository / "platforms/aspire2ap/run-qwen-pd.pbs").read_text()
    checkers = [
        (repository / "platforms/genkai/check-qwen-job.sh").read_text(),
        (repository / "platforms/aspire2ap/check-qwen-job.sh").read_text(),
        (repository / "platforms/firmus/check-qwen-job.sh").read_text(),
    ]

    for workflow, checker in zip((genkai, aspire, firmus), checkers, strict=True):
        assert "monitor-qwen-job.sh" in workflow
        assert "profile_export_aiperf.json" in checker
        assert "score.json" in checker
        assert "upload-artifact@v7" in workflow
        assert "configure-aws-credentials" not in workflow
    assert "provision-qwen-pd.sh" in genkai
    assert "provision-qwen-pd.sh" in aspire
    assert "provision-firmus-framework.sh" in firmus
    assert '--mpi "proc=${{ inputs.node_count }}"' in genkai
    assert "module load intel/2025.1.3 impi" in genkai_job
    assert "QWEN_PLATFORM=genkai" in genkai_job
    assert "GPUS_PER_NODE=4" in genkai_job
    assert "select=2:ncpus=112:ngpus=8:mem=1500gb:mpiprocs=1" in aspire
    assert "place=scatter:excl" in aspire
    assert "QWEN_PLATFORM=aspire2ap" in aspire_job
    assert "GPUS_PER_NODE=8" in aspire_job
    assert "/opt/pbs/bin/pbsdsh" in submission
    assert 'mpirun -np "${NODE_COUNT}" -ppn 1' in submission
    assert "GOOGLE_CREDENTIAL" not in submission
    assert "AWS_ACCESS_KEY" not in submission


def test_cluster_topologies_use_every_allocated_gpu() -> None:
    repository = Path(__file__).parents[2]
    submission = (repository / "submission/run/submit.sbatch").read_text()

    assert 'firmus) [[ "${NODE_COUNT}" -eq 2 && "${GPUS_PER_NODE}" -eq 8 ]]' in submission
    assert 'aspire2ap) [[ "${NODE_COUNT}" -eq 2 && "${GPUS_PER_NODE}" -eq 8 ]]' in submission
    assert 'genkai) [[ "${GPUS_PER_NODE}" -eq 4' in submission
    assert ".workers[] | select(.node==$node)" in submission
    config = json.loads((repository / "configs/aspire-serving.json").read_text())
    assert [(w["role"], w["node"], w["tp"], w["ep"], w["gpus"]) for w in config["workers"]] == [("prefill", 0, 4, 4, "0,1,2,3"), ("prefill", 0, 4, 4, "4,5,6,7"), ("decode", 1, 4, 4, "0,1,2,3"), ("decode", 1, 4, 4, "4,5,6,7")]
    assert 'start_worker prefill "${node_rank}" 4' in submission
    assert 'start_worker decode "${decode_index}" 4' in submission
