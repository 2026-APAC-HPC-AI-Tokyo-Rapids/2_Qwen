"""One CPU-only prediction for the configuration in this commit, from repo root."""

import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys

import yaml
from aisimulate_core.sdk import RustForwardPassPerfModel


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    output = pathlib.Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    config_path = pathlib.Path("research/simulation/prediction.yaml")
    config = yaml.safe_load(config_path.read_text())
    engine = config["engine"]
    model = pathlib.Path(engine["model"]) / "config.json"
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    assert model_sha == "f056921e486d3e22aed125b5218b87146ea2ac838ccf1521dea5b08f809e8d59"
    assert engine["mode"] == "disaggregated"
    workers = engine["workers"]
    assert sum(w["parallelism"]["replicas"] * w["parallelism"]["tensor"] * w["parallelism"]["pipeline"] for w in workers.values()) == 16
    write_json(output / "provenance.json", {
        "kind": "simulation_not_official_result",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "aisimulate": importlib.metadata.version("aisimulate"),
        "python": sys.version,
        "model_revision": "e156cb4efae43fbee1a1ab073f946a1377e6b969",
        "model_config_sha256": model_sha,
        "prediction_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "assumptions": ["Generic H100 SXM data, not ASPIRE calibration", "SGLang 0.5.14 performance database, not the deployed source commit", "50 GB/s assumed aggregate KV transfer rate", "Synthetic lengths only; no official AIPerf warmup or measurement window", "Engine routing is not sglang-router 0.3.2"],
    })
    (output / "prediction.yaml").write_text(config_path.read_text())
    diagnostics = {}
    for role, worker in workers.items():
        shape = worker["parallelism"]
        request = {"model": engine["model"], "system": engine["hardware"], "backend": engine["backend"], "backend_version": engine["backend_version"], "worker_type": role, "tp": shape["tensor"], "pp": shape["pipeline"], "attention_dp": shape["attention_data"], "moe_tp_size": shape["moe_tensor"], "moe_ep_size": shape["moe_expert"], "estimation_mode": "op_level", "fallback_policy": "deny"}
        try:
            estimator = RustForwardPassPerfModel.best_available(request)
            try:
                diagnostics[role] = estimator.diagnostics()
            finally:
                estimator.close()
        except ValueError as error:
            diagnostics[role] = {"readiness": "unsupported_config", "error": str(error)}
    write_json(output / "estimator-diagnostics.json", diagnostics)
    if any(d.get("readiness") != "ready" for d in diagnostics.values()):
        write_json(output / "status.json", {"status": "unsupported_configuration", "official_result": False})
        print(json.dumps(diagnostics, indent=2))
        return 2
    command = [str(pathlib.Path(sys.executable).with_name("aisimulate")), "predict", "--config", str(config_path), "--stack", "engine", "--output-dir", str(output / "prediction"), "--capture-per-request", "--detail", "all"]
    with (output / "predict.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        write_json(output / "status.json", {"status": "prediction_failed", "exit_code": result.returncode, "official_result": False})
        print((output / "predict.log").read_text())
        return result.returncode
    report = json.loads((output / "prediction/prediction.json").read_text())
    requests = [json.loads(line) for line in (output / "prediction/requests.jsonl").read_text().splitlines()]
    assert len(requests) == config["traffic"]["stop"]["requests"]
    good = sum(r["terminal_status"] == "completed" and r["ttft_ms"] <= 3000 and r["itl_ms"] <= 20 for r in requests)
    summary = {
        "status": "prediction_completed", "official_result": False,
        "simulated_requests": len(requests), "simulated_good_requests": good,
        "simulated_joint_sla_fraction": good / len(requests),
        "simulated_joint_sla_gate": good / len(requests) >= 0.90,
        "metrics": {k: report[k] for k in ("duration_ms", "request_throughput_rps", "output_throughput_tok_s", "mean_ttft_ms", "p99_ttft_ms", "mean_itl_ms", "p99_itl_ms", "wall_time_ms")},
    }
    write_json(output / "status.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
