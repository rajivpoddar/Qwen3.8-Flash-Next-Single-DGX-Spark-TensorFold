"""Host-side bounded maintenance runner; always restore the untouched live container."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

SOURCE = "fbb271cc102f4fa8c5075bc9368c22c0dba5452671456e928e92002b46d3b6ef"
IMAGE = "sha256:2afbaf2c008834e645f11dfa512d5504f1889142ea6849efabe92b98a64ef2ef"
BENCH = "qwen38-mtp-sweep-20260930-matched"
ROOT = Path("/home/user/serve/tensorfold-spark-30000/spark")
RECEIPT = ROOT / "mtp-sweep-maintenance-20260930-matched.json"


def output(*cmd):
    return subprocess.check_output(cmd, text=True).strip()


def inspect(target):
    return json.loads(output("docker", "inspect", target))[0]


source = inspect("qwen38-flash-next-tf")
assert source["Id"] == SOURCE and source["Image"] == IMAGE
assert source["State"]["Running"] and not source["State"]["OOMKilled"]
assert BENCH not in output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=5) as response:
    assert json.load(response)["requests_running"] == 0, "live engine not drained"
host = source["HostConfig"]
assert host["IpcMode"] == host["NetworkMode"] == "host"
assert host["RestartPolicy"]["Name"] == "unless-stopped"
command = ["docker", "run", "-d", "--name", BENCH, "--gpus", "all", "--ipc=host", "--network=host"]
for limit in host.get("Ulimits") or []:
    command += ["--ulimit", f'{limit["Name"]}={limit["Soft"]}:{limit["Hard"]}']
for entry in source["Config"]["Env"]:
    command += ["-e", entry]
for mount in source["Mounts"]:
    assert mount["Type"] == "bind"
    command += ["-v", f'{mount["Source"]}:{mount["Destination"]}:{"rw" if mount["RW"] else "ro"}']
command += ["-v", f"{ROOT / 'mtp_sweep.py'}:/opt/mtp_sweep.py:ro", IMAGE, "python", "-u", "/opt/mtp_sweep.py"]
receipt = {"source_id": SOURCE, "image_id": IMAGE, "source_cmd": source["Config"]["Cmd"],
           "benchmark": BENCH, "started_at": time.time(), "client_contract": "unchanged",
           "mop": "untouched per user directive", "state": "starting"}
try:
    output("docker", "stop", "--time", "30", SOURCE)
    assert not inspect(SOURCE)["State"]["Running"]
    receipt["benchmark_id"] = output(*command)
    receipt["state"] = "benchmark_running"
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    deadline = time.monotonic() + 900
    while inspect(BENCH)["State"]["Running"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded sweep exceeded 15 minutes; restoring live backend")
        time.sleep(2)
    state = inspect(BENCH)["State"]
    receipt["benchmark_exit"] = state["ExitCode"]
    receipt["benchmark_oom"] = state["OOMKilled"]
    assert state["ExitCode"] == 0 and not state["OOMKilled"], "sweep failed; inspect preserved benchmark logs"
    receipt["state"] = "benchmark_complete"
finally:
    if BENCH in output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        output("docker", "stop", "--time", "10", BENCH)
    output("docker", "start", SOURCE)
    assert inspect(SOURCE)["Config"]["Cmd"] == source["Config"]["Cmd"]
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8888/v1/models", timeout=5) as response:
                models = json.load(response)
            assert models["data"][0]["id"] == "qwen3.8-flash-next"
            receipt["source_restored"] = True
            break
        except Exception:
            time.sleep(2)
    receipt["finished_at"] = time.time()
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    assert receipt.get("source_restored"), "restored source has not reached model discovery"
