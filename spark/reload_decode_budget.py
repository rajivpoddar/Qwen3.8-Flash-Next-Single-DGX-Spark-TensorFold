"""Replace only the pinned TensorFold backend, preserving its stopped container.

No gateway, dashboard, MoP, slot process, model flag, mount, or credential changes.
Run remotely after affected request turns are interrupted and the engine is idle.
"""
import json
import subprocess
import urllib.request

SOURCE_ID = "aec6f7782752ff9a42124f27b4d08a82f60c1eeb4704f02a4b9f7e913070d10d"
SOURCE_IMAGE = "sha256:c3a5ff2ba93e57419e0b34d6b544f86d6b4883e8eed37fd38ee3a424a1ccda87"
CANDIDATE_IMAGE = "tensorfold-qwen38:v0.3.6.3-decode-budget"
NAME = "qwen38-flash-next-tf"
ROLLBACK = "qwen38-flash-next-tf-before-decode-budget-20260930"


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def inspect(target):
    return json.loads(output("docker", "inspect", target))[0]


def mount_tuple(container):
    # Docker represents an implicit rw mount as Mode="", an explicit one as "rw".
    return sorted((m["Type"], m["Source"], m["Destination"], m["RW"], m["Propagation"])
                  for m in container["Mounts"])


source = inspect(NAME)
assert source["Id"] == SOURCE_ID and source["Image"] == SOURCE_IMAGE, "source identity drift"
assert source["State"]["Running"] and not source["State"]["OOMKilled"]
assert source["Config"]["Cmd"][0:3] == ["tensorfold", "serve", "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"]
assert output("docker", "image", "inspect", "--format", "{{index .Config.Labels \"spark.prefill_scheduling\"}}",
              CANDIDATE_IMAGE) == "bounded-decode-share"
names = output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
assert ROLLBACK not in names, "rollback name already exists"
with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=5) as response:
    assert json.load(response)["requests_running"] == 0, "engine not drained"

host = source["HostConfig"]
assert host["IpcMode"] == host["NetworkMode"] == "host"
assert host["RestartPolicy"]["Name"] == "unless-stopped"
assert host["Memory"] == 0 and host["DeviceRequests"][0]["Count"] == -1
cmd = ["docker", "run", "-d", "--name", NAME, "--gpus", "all", "--ipc=host", "--network=host",
       "--restart=unless-stopped"]
for limit in host.get("Ulimits") or []:
    cmd.extend(["--ulimit", f'{limit["Name"]}={limit["Soft"]}:{limit["Hard"]}'])
environment = {entry.split("=", 1)[0]: entry.split("=", 1)[1] for entry in source["Config"]["Env"]}
environment["TENSORFOLD_PREFILL_DECODE_SHARE"] = "0.20"
for key, value in environment.items():
    cmd.extend(["-e", f"{key}={value}"])
for mount in source["Mounts"]:
    assert mount["Type"] == "bind", "unexpected mount type"
    mode = "rw" if mount["RW"] else "ro"
    cmd.extend(["-v", f'{mount["Source"]}:{mount["Destination"]}:{mode}'])
cmd.extend([CANDIDATE_IMAGE, *source["Config"]["Cmd"]])

output("docker", "stop", "--time", "30", SOURCE_ID)
output("docker", "rename", SOURCE_ID, ROLLBACK)
try:
    candidate_id = output(*cmd)  # command/environment deliberately never printed
except Exception:
    if NAME in output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        output("docker", "rename", NAME, NAME + "-failed-decode-budget-20260930")
    output("docker", "rename", SOURCE_ID, NAME)
    output("docker", "start", SOURCE_ID)
    raise

candidate = inspect(candidate_id)
assert candidate["Config"]["Cmd"] == source["Config"]["Cmd"]
assert mount_tuple(candidate) == mount_tuple(source)
assert candidate["HostConfig"]["Ulimits"] == host["Ulimits"]
assert candidate["State"]["Running"]
print(json.dumps({"candidate_id": candidate_id, "image_id": candidate["Image"],
                  "rollback": ROLLBACK, "rollback_id": SOURCE_ID, "decode_share": 0.20,
                  "client_contract": "unchanged", "flags_mounts_ulimits": "identical"}), flush=True)
