"""Activate the retained, already-benchmarked candidate after an ordering-only check rollback."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

SOURCE = "fbb271cc102f4fa8c5075bc9368c22c0dba5452671456e928e92002b46d3b6ef"
CANDIDATE = "fc0d01efb31bd803432d61e54404334029606d7a676dc1b7af9fc858b60be3d7"
IMAGE = "sha256:3bf7dc3341a116b3a226338303f4bf2974080e6d68dbea96cdcf4c711df0a879"
NAME = "qwen38-flash-next-tf"
ROLLBACK = NAME + "-before-prefix-retention-20261001"
FAILED = NAME + "-failed-prefix-retention-20261001"
ROOT = Path("/home/user/serve/tensorfold-spark-30000/spark")
CACHE = Path("/home/user/.cache/tensorfold-qwen38")
RECEIPT = CACHE / "prefix-retention-activation-20261001.json"


def output(*cmd):
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} operation failed with exit {result.returncode}")
    return result.stdout.strip()


def inspect(target):
    return json.loads(output("docker", "inspect", target))[0]


def wait_ready():
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8888/v1/models", timeout=5) as response:
                assert json.load(response)["data"][0]["id"] == "qwen3.8-flash-next"
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("model discovery did not become ready in five minutes")


source, candidate = inspect(SOURCE), inspect(CANDIDATE)
assert source["Name"] == "/" + NAME and source["State"]["Running"]
assert source["Image"] == "sha256:2afbaf2c008834e645f11dfa512d5504f1889142ea6849efabe92b98a64ef2ef"
assert candidate["Name"] == "/" + FAILED and not candidate["State"]["Running"]
assert candidate["Image"] == IMAGE and not candidate["State"]["OOMKilled"]
report = json.loads((CACHE / "mtp-sweep-20261001-prefix-retention.json").read_text())
assert report["status"] == "complete" and report["output_parity"]
assert report["recommended_setting"] == [6, .6] and len(report["cache_preflight"]) == 2
assert candidate["Config"]["Cmd"] == source["Config"]["Cmd"]
env = lambda c: dict(e.split("=", 1) for e in c["Config"]["Env"])
assert env(candidate) == env(source), "environment values differ"
mounts = lambda c: sorted((m["Type"], m["Source"], m["Destination"], m["RW"], m["Propagation"])
                          for m in c["Mounts"])
assert mounts(candidate) == mounts(source)
for field in ("Ulimits", "RestartPolicy", "DeviceRequests", "IpcMode", "NetworkMode", "Memory"):
    assert candidate["HostConfig"][field] == source["HostConfig"][field], field
assert ROLLBACK not in output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=5) as response:
    assert json.load(response)["requests_running"] == 0

receipt = {"source_id": SOURCE, "candidate_id": CANDIDATE, "image_id": IMAGE,
           "rollback": ROLLBACK, "cmd": candidate["Config"]["Cmd"], "started_at": time.time(),
           "environment_values": "identical", "ordering_only_check_corrected": True,
           "client_contract": "unchanged", "mop": "untouched", "state": "starting"}


def save(state, **fields):
    receipt.update(state=state, updated_at=time.time(), **fields)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"state": state, **fields}), flush=True)


success = False
try:
    output("docker", "stop", "--time", "30", SOURCE)
    output("docker", "rename", SOURCE, ROLLBACK)
    output("docker", "rename", CANDIDATE, NAME)
    output("docker", "start", CANDIDATE)
    save("candidate_loading")
    wait_ready()
    save("model_ready")
    output("bash", str(ROOT / "prove-stack.sh"))
    for script, artifact in (("prove_identical_cache.py", "prefix-retention-api-proof-20261001.json"),
                             ("prove_tool_prefix_cache.py", "prefix-retention-tool-proof-20261001.json")):
        proof = output("python3", str(ROOT / script))
        CACHE.joinpath(artifact).write_text(proof + "\n")
        save("proof_pass", proof=script)
    final = inspect(CANDIDATE)
    assert final["State"]["Running"] and not final["State"]["OOMKilled"] and final["RestartCount"] == 0
    success = True
    save("complete", authenticated_readiness=True, persistent_api_cache=True, tool_parity=True,
         slot_restarts="none", rehandoff="none", finished_at=time.time())
finally:
    if not success:
        output("docker", "stop", "--time", "10", CANDIDATE)
        if inspect(CANDIDATE)["Name"] != "/" + FAILED:
            output("docker", "rename", CANDIDATE, FAILED)
        if inspect(SOURCE)["Name"] != "/" + NAME:
            output("docker", "rename", SOURCE, NAME)
        output("docker", "start", SOURCE)
        wait_ready()
        output("bash", str(ROOT / "prove-stack.sh"))
        save("rolled_back", source_restored=True, finished_at=time.time())
