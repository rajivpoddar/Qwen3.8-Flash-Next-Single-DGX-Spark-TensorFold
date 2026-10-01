"""Bounded real-model copy/MTP/serial parity through the existing private API.

The four simultaneous seeded requests mix quoting with novel output. Baseline
and candidate use the same sampling; copy counters must prove actual verified
and accepted copied tokens. No external tools or customer data are used.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--reference")
args = parser.parse_args()
BASE = "http://127.0.0.1:8888"


def call(path, body=None, timeout=240):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


CODE = '''def reconcile_rows(original, updates):
    """Return a new list, preserving every unchanged row and its original order."""
    result = []
    for index, row in enumerate(original):
        if index in updates:
            result.append(updates[index])
        else:
            result.append(row)
    return result

def summarize_rows(rows):
    total = 0
    for row in rows:
        total += len(row)
    return {"rows": len(rows), "characters": total}
'''
CASES = [
    ("copy_greedy", "Repeat the following Python code exactly, without commentary. Preserve every line:\n" + CODE, 0),
    ("copy_sampled", "Output exactly the following Python code. Do not change or explain it:\n" + CODE, .6),
    ("novel_greedy", "Write a Python implementation of binary search, then explain its correctness and complexity.", 0),
    ("novel_sampled", "Write a Python implementation of a bounded least-recently-used cache. Explain its invariants.", .6),
]


def generate(case, draft=True, barrier=None):
    name, prompt, temperature = case
    if barrier is not None:
        barrier.wait(timeout=10)
    started = time.monotonic()
    response = call("/v1/chat/completions", {"model": "qwen3.8-flash-next", "temperature": temperature,
        "top_p": .95, "top_k": 20, "seed": 1234, "max_tokens": 384, "draft": draft,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}]})
    assert response["choices"][0]["message"].get("content"), response
    return {"case": name, "draft": draft, "elapsed_s": round(time.monotonic() - started, 4),
            "tensorfold": response["tensorfold"], "usage": response["usage"],
            "content_sha256": hashlib.sha256(response["choices"][0]["message"]["content"].encode()).hexdigest()}


barrier, samples = threading.Barrier(4), []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(generate, case, True, barrier) for case in CASES]
    deadline = time.monotonic() + 240
    while not all(f.done() for f in futures):
        assert time.monotonic() < deadline, "copy proof timed out"
        samples.append(call("/health", timeout=5))
        time.sleep(.25)
    results = [f.result() for f in futures]
proof = {"status": "running", "results": results,
         "peak_decoding": max(h["streams"]["decoding"] for h in samples)}
Path(args.output).write_text(json.dumps(proof, indent=2) + "\n")
assert proof["peak_decoding"] == 4, proof
if args.reference:
    reference = json.loads(Path(args.reference).read_text())
    for current, old in zip(results, reference["results"]):
        assert current["case"] == old["case"]
        assert current["tensorfold"]["token_sha"] == old["tensorfold"]["token_sha"], current["case"]
    proof["baseline_token_parity"] = True
    proof["copy_drafted"] = sum(r["tensorfold"].get("copy_drafted", 0) for r in results)
    proof["copy_accepted"] = sum(r["tensorfold"].get("copy_accepted", 0) for r in results)
    assert proof["copy_drafted"] > 0 and proof["copy_accepted"] > 0, "no accepted prompt-copy drafts"
    proof["serial"] = []
    for case, drafted in zip(CASES, results):
        serial = generate(case, draft=False)
        proof["serial"].append(serial)
        assert serial["tensorfold"]["token_sha"] == drafted["tensorfold"]["token_sha"], case[0]
        assert serial["tensorfold"].get("copy_drafted", 0) == 0
    proof["serial_token_parity"] = True
state = call("/health", timeout=5)
assert state["requests_running"] == 0 and state["context_length"] == 262144 and state["streams"]["max"] == 4
proof.update(status="complete", final_health=state)
Path(args.output).write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps({k: v for k, v in proof.items() if k not in {"results", "serial"}}), flush=True)
