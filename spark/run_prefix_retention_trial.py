"""One bounded sweep/load window with exact-container automatic rollback.

Never prints launch environments or credentials. The prior backend is retained,
and gateway/dashboard/MoP/Claude processes are outside this runner's scope.
"""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

SOURCE = "fbb271cc102f4fa8c5075bc9368c22c0dba5452671456e928e92002b46d3b6ef"
OLD_IMAGE = "sha256:2afbaf2c008834e645f11dfa512d5504f1889142ea6849efabe92b98a64ef2ef"
IMAGE = "sha256:3bf7dc3341a116b3a226338303f4bf2974080e6d68dbea96cdcf4c711df0a879"
NAME = "qwen38-flash-next-tf"
ROLLBACK = NAME + "-before-prefix-retention-20261001"
FAILED = NAME + "-failed-prefix-retention-20261001"
BENCH = "qwen38-mtp-sweep-20261001-prefix-retention"
ROOT = Path("/home/user/serve/tensorfold-spark-30000/spark")
CACHE = Path("/home/user/.cache/tensorfold-qwen38")
RESULT = CACHE / "mtp-sweep-20261001-prefix-retention.json"
RECEIPT = CACHE / "prefix-retention-maintenance-20261001.json"


def output(*cmd):
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        # CalledProcessError would expose the entire launch command/environment.
        raise RuntimeError(f"{cmd[0]} operation failed with exit {result.returncode}")
    return result.stdout.strip()


def inspect(target):
    return json.loads(output("docker", "inspect", target))[0]


def names():
    return output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()


def discovery():
    with urllib.request.urlopen("http://127.0.0.1:8888/v1/models", timeout=5) as response:
        assert json.load(response)["data"][0]["id"] == "qwen3.8-flash-next"


def wait_ready():
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            discovery()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("backend model discovery did not become ready in five minutes")


def mount_tuple(c):
    return sorted((m["Type"], m["Source"], m["Destination"], m["RW"], m["Propagation"])
                  for m in c["Mounts"])


source = inspect(NAME)
assert source["Id"] == SOURCE and source["Image"] == OLD_IMAGE, "source identity drift"
assert source["State"]["Running"] and not source["State"]["OOMKilled"]
assert not any(n in names() for n in (ROLLBACK, FAILED, BENCH)), "trial target already exists"
assert output("docker", "image", "inspect", "--format",
              '{{index .Config.Labels "spark.prefix_retention"}}', IMAGE) == "reuse-tail-checkpoint"
assert source["Config"]["Cmd"][0:3] == ["tensorfold", "serve", "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"]
with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=5) as response:
    assert json.load(response)["requests_running"] == 0, "live engine not drained"
host = source["HostConfig"]
assert host["IpcMode"] == host["NetworkMode"] == "host"
assert host["RestartPolicy"]["Name"] == "unless-stopped"
assert host["Memory"] == 0 and host["DeviceRequests"][0]["Count"] == -1


def launch(name, command, *, benchmark=False):
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--ipc=host", "--network=host"]
    if not benchmark:
        cmd += ["--restart=unless-stopped"]
    for limit in host.get("Ulimits") or []:
        cmd += ["--ulimit", f'{limit["Name"]}={limit["Soft"]}:{limit["Hard"]}']
    for entry in source["Config"]["Env"]:
        cmd += ["-e", entry]
    for mount in source["Mounts"]:
        assert mount["Type"] == "bind"
        cmd += ["-v", f'{mount["Source"]}:{mount["Destination"]}:{"rw" if mount["RW"] else "ro"}']
    if benchmark:
        cmd += ["-v", f"{ROOT / 'mtp_sweep.py'}:/opt/mtp_sweep.py:ro"]
    return output(*cmd, IMAGE, *command)


receipt = {"source_id": SOURCE, "source_image": OLD_IMAGE, "candidate_image": IMAGE,
           "source_cmd": source["Config"]["Cmd"], "rollback": ROLLBACK,
           "started_at": time.time(), "client_contract": "unchanged", "mop": "untouched",
           "state": "starting"}


def save(state, **fields):
    receipt.update(state=state, updated_at=time.time(), **fields)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"state": state, **fields}), flush=True)


success = False
try:
    output("docker", "stop", "--time", "30", SOURCE)
    assert not inspect(SOURCE)["State"]["Running"]
    bench_id = launch(BENCH, ["python", "-u", "/opt/mtp_sweep.py"], benchmark=True)
    save("benchmark_running", benchmark_id=bench_id)
    deadline = time.monotonic() + 900
    while inspect(BENCH)["State"]["Running"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("sweep exceeded fifteen minutes")
        time.sleep(2)
    state = inspect(BENCH)["State"]
    assert state["ExitCode"] == 0 and not state["OOMKilled"], "sweep failed; retained logs contain the cause"
    report = json.loads(RESULT.read_text())
    assert report["status"] == "complete" and report["output_parity"]
    assert len(report["cache_preflight"]) == 2
    depth, confidence = report["recommended_setting"]
    assert depth in (3, 4, 6) and confidence in (.30, .60)
    command = list(source["Config"]["Cmd"])
    command[command.index("--mtp-drafts") + 1] = str(depth)
    command[command.index("--mtp-confidence") + 1] = f"{confidence:.2f}"
    save("benchmark_complete", recommended_setting=[depth, confidence])
    output("docker", "rename", SOURCE, ROLLBACK)
    candidate_id = launch(NAME, command)
    save("candidate_loading", candidate_id=candidate_id, candidate_cmd=command)
    candidate = inspect(candidate_id)
    assert candidate["Image"] == IMAGE and candidate["Config"]["Cmd"] == command
    assert dict(e.split("=", 1) for e in candidate["Config"]["Env"]) == \
        dict(e.split("=", 1) for e in source["Config"]["Env"])
    assert mount_tuple(candidate) == mount_tuple(source)
    assert candidate["HostConfig"]["Ulimits"] == host["Ulimits"]
    wait_ready()
    output("bash", str(ROOT / "prove-stack.sh"))
    cache_proof = output("python3", str(ROOT / "prove_identical_cache.py"))
    CACHE.joinpath("prefix-retention-api-proof-20261001.json").write_text(cache_proof + "\n")
    candidate = inspect(candidate_id)
    assert candidate["State"]["Running"] and not candidate["State"]["OOMKilled"]
    assert candidate["RestartCount"] == 0
    success = True
    save("complete", authenticated_readiness=True, persistent_api_cache=True,
         finished_at=time.time(), slot_restarts="none", rehandoff="none")
finally:
    if not success:
        if BENCH in names() and inspect(BENCH)["State"]["Running"]:
            output("docker", "stop", "--time", "10", BENCH)
        if NAME in names() and inspect(NAME)["Id"] != SOURCE:
            output("docker", "stop", "--time", "10", NAME)
            output("docker", "rename", NAME, FAILED)
        if inspect(SOURCE)["Name"] != "/" + NAME:
            output("docker", "rename", SOURCE, NAME)
        output("docker", "start", SOURCE)
        wait_ready()
        output("bash", str(ROOT / "prove-stack.sh"))
        save("rolled_back", source_restored=True, finished_at=time.time())
