"""Bounded MTP comparison on one resident copy of the exact serving engine.

Maintenance-only harness, not an HTTP endpoint or a production modification.
All four full-context buffers and the vision tower remain allocated. Parameters
change only after the scheduler has drained; depths never exceed their allocation.
"""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from tensorfold.cuda.server import App
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

RESULT = Path("/cache/mtp-sweep-20261001-prefix-retention.json")
MODEL = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
SETTINGS = [(6, .60), (3, .30), (4, .60), (6, .30), (3, .60), (4, .30)]
CODE = ("Write a Python module implementing an immutable interval set. Implement union, "
        "intersection and subtraction with half-open intervals, explain invariants in docstrings, "
        "and include boundary and randomized unit tests. Do not quote the background notes.")
FILLER = "alpha beta gamma delta epsilon zeta eta theta\n" * 6500
report = {"model": MODEL, "settings": SETTINGS, "context": 262144, "streams": 4,
          "kv": "int4", "prefill_rows": 2048, "decode_share": .20,
          "tokens_per_reply_cap": 256, "thinking_enabled_all_cases": True,
          "workload_labels": {"code": "greedy coding with thinking", "thinking": "sampled coding with thinking"},
          "weights_loaded": 1, "samples": [],
          "references": {}, "status": "starting", "started_at": time.time()}


def save(event, **fields):
    report["status"] = event
    report["updated_at"] = time.time()
    RESULT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": event, **fields}), flush=True)


def knobs(engine, depth, confidence):
    assert engine.multi.live() == 0 and engine.scheduler.waiting.empty()
    assert not engine.scheduler.boxes, "scheduler replies still pending"
    assert depth in (3, 4, 6) and confidence in (.30, .60)
    assert engine.multi.buf.rows == 28, "full four-stream buffers changed"
    engine.depth = engine.multi.depth = depth
    engine.confidence = engine.multi.confidence = confidence


def body(kind, slot, long, tokens=256):
    # Keep the rendered prompt identical so every arm reuses the same checkpoint.
    # Sampling, not
    # the prompt or thinking flag, distinguishes these two workload cases.
    question = CODE
    messages = [{"role": "system", "content": f"Synthetic benchmark stream {slot}. "
                 "Ignore the background notes; answer the user's engineering request.\n" +
                 (FILLER if long else "")}, {"role": "user", "content": question}]
    return {"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": tokens,
            "temperature": 0 if kind == "code" else 1., "top_k": 20, "top_p": .95,
            "seed": 1234 + slot, "chat_template_kwargs": {"enable_thinking": True}}


def one(app, request, draft=True):
    prepared = app.prepare(request, chat=True)
    output = []
    start = time.perf_counter()
    first = [None]

    def emit(tokens):
        if first[0] is None:
            first[0] = time.perf_counter()
        output.extend(tokens)
        return False

    stats = app.engine.generate(prepared.prompt, prepared.max_tokens, prepared.sampling, emit, draft=draft)
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256(json.dumps(output, separators=(",", ":")).encode()).hexdigest()
    return {"prompt_tokens": len(prepared.prompt), "completion_tokens": len(output),
            "token_sha": digest, "elapsed_s": elapsed, "ttft_s": first[0] - start,
            "decode_tps": max(0, len(output) - 1) / max(stats["decode_s"], .0001),
            **stats}


def group(app, kind, count, long, tokens=256, draft=True):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(one, app, body(kind, slot, long, tokens), draft) for slot in range(count)]
        rows = [future.result(timeout=300) for future in futures]
    elapsed = time.perf_counter() - start
    drain(app)
    return {"kind": kind, "concurrency": count, "long_context": long, "elapsed_s": elapsed,
            "median_per_request_tps": statistics.median(row["decode_tps"] for row in rows),
            "minimum_per_request_tps": min(row["decode_tps"] for row in rows),
            "batch_end_to_end_tps": sum(row["completion_tokens"] for row in rows) / elapsed,
            "accepted": sum(row["accepted"] for row in rows),
            "drafted": sum(row["drafted"] for row in rows),
            "tokens_per_round": sum(max(0, row["completion_tokens"] - 1) for row in rows) /
                                max(1, sum(row["rounds"] for row in rows)),
            "rows": rows, "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30}


def drain(app):
    torch.cuda.synchronize()
    # Completion can arrive just before the scheduler removes its box.
    until = time.monotonic() + 5
    while app.engine.scheduler.boxes or app.engine.multi.live():
        assert time.monotonic() < until, "scheduler did not drain"
        time.sleep(.01)


def cache_preflight(app):
    """Real model proof before the larger trial: repeated warm hits and serial parity."""
    cases = []
    for kind in ("code", "thinking"):
        request = body(kind, 0, False, tokens=16)
        request["messages"][0]["content"] += "\n" + "alpha beta gamma delta epsilon zeta eta theta\n" * 800
        rows = []
        for repeat in range(4):
            rows.append(one(app, request))
            drain(app)
            if repeat:
                assert rows[-1]["cached"] >= 6144, "repeated request lost its usable prefix"
        reference = one(app, request, draft=False)
        drain(app)
        assert all(row["token_sha"] == reference["token_sha"] for row in rows), "reused snapshot changed output"
        cases.append({"kind": kind, "rows": rows, "serial": reference})
        save("cache_preflight_pass", kind=kind, cached=[row["cached"] for row in rows],
             prefill_s=[row["prefill_s"] for row in rows], serial_parity=True)
    report["cache_preflight"] = cases


def checked_sample(app, depth, confidence, kind, count, long, phase):
    knobs(app.engine, depth, confidence)
    sample = group(app, kind, count, long)
    sample.update(depth=depth, confidence=confidence, phase=phase)
    key = f"{kind}:{count}:{long}"
    hashes = [row["token_sha"] for row in sample["rows"]]
    if key not in report["references"]:
        report["references"][key] = hashes
    assert hashes == report["references"][key], f"output changed in {key} at {depth}/{confidence}"
    if long:
        assert all(row["cached"] >= 60000 for row in sample["rows"]), "unmatched cold-prefill timing"
    report["samples"].append(sample)
    save("sample", depth=depth, confidence=confidence, kind=kind, concurrency=count,
         long_context=long, phase=phase, per_request_tps=sample["median_per_request_tps"],
         accepted=sample["accepted"], drafted=sample["drafted"], output_parity=True)


model_dir = Path(snapshot_download(MODEL, local_files_only=True))
if sys.argv[1:] == ["--check-fixtures"]:
    check_app = App(SimpleNamespace(context_window=262144), model_dir,
                    "qwen3.8-flash-next", default_thinking=True, context_window=262144)
    prefixes = []
    for slot in range(4):
        a = check_app.prepare(body("code", slot, True), chat=True).prompt
        b = check_app.prepare(body("thinking", slot, True), chat=True).prompt
        common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        assert a == b, "fixtures must have identical rendered tokens for any retained boundary"
        prefixes.append({"slot": slot, "code_tokens": len(a), "thinking_tokens": len(b),
                         "common_prefix_tokens": common})
    print(json.dumps({"fixture_check": "PASS", "prefixes": prefixes}), flush=True)
    raise SystemExit(0)
save("loading")
report["snapshot"] = str(model_dir)
engine = FlashNextEngine(model_dir, depth=6, confidence=.60, max_len=262144,
                         context_explicit=True, streams=4, ple_on_ssd=True,
                         kv_dtype="int4", vision=True)
app = App(engine, model_dir, "qwen3.8-flash-next", default_thinking=True, context_window=262144)
report["capacity"] = engine.capacity_plan
save("loaded", cache_slots=engine.max_len, streams=4)

cache_preflight(app)

# Cold prefill is excluded from the sweep. Four distinct prefixes warm every slot.
warm = group(app, "code", 4, True, tokens=16)
report["cold_warmup"] = warm
save("long_prefixes_warmed", prompt_tokens=[row["prompt_tokens"] for row in warm["rows"]])

# Long-context samples go first; the short samples use only one idle slot and
# would otherwise evict the long prefix. Re-warm that one slot before the next arm.
for depth, confidence in SETTINGS:
    for kind in ("code", "thinking"):
        checked_sample(app, depth, confidence, kind, 4, True, "screen")

def screen_score(setting):
    rows = [s for s in report["samples"] if s["phase"] == "screen" and
            (s["depth"], s["confidence"]) == setting]
    return math.exp(statistics.mean(math.log(s["median_per_request_tps"]) for s in rows))

best = max(SETTINGS, key=screen_score)
report["screen_winner"] = list(best)
# Alternate baseline and provisional winner. Require the median repeat gain to
# exceed 5% without a >5% regression in either workload before recommending it.
repeat_settings = [(6, .60), best, (6, .60), best] if best != (6, .60) else [(6, .60)]
for depth, confidence in repeat_settings:
    for kind in ("code", "thinking"):
        checked_sample(app, depth, confidence, kind, 4, True, "confirm")

# Solo fixtures use the same eager four-stream engine, not the faster serial
# CUDA-graph engine. Long cached prefixes are no longer needed after this phase.
for depth, confidence in SETTINGS:
    for kind in ("code", "thinking"):
        checked_sample(app, depth, confidence, kind, 1, False, "solo")

# Independent uncached, no-draft references for both short and long fixtures.
knobs(engine, 6, .60)
serial = []
for long in (False, True):
    for kind in ("code", "thinking"):
        row = one(app, body(kind, 0, long), draft=False)
        key = f"{kind}:{4 if long else 1}:{long}"
        assert row["token_sha"] == report["references"][key][0], "uncached serial mismatch"
        serial.append({"kind": kind, "long_context": long, **row})
        save("serial_reference_pass", kind=kind, long_context=long)
report["serial_references"] = serial

gains = {}
for kind in ("code", "thinking"):
    def rate(setting):
        return statistics.median(s["median_per_request_tps"] for s in report["samples"]
                                 if s["phase"] == "confirm" and s["kind"] == kind and
                                 (s["depth"], s["confidence"]) == setting)
    gains[kind] = rate(best) / rate((6, .60)) if best != (6, .60) else 1.
combined = math.sqrt(gains["code"] * gains["thinking"])
report["confirmed_gains"] = gains
report["confirmed_combined_gain"] = combined
report["recommended_setting"] = list(best if combined >= 1.05 and min(gains.values()) >= .95 else (6, .60))
report["output_parity"] = True
report["duration_s"] = time.time() - report["started_at"]
save("complete", recommended_setting=report["recommended_setting"], gains=gains,
     combined_gain=combined, duration_s=report["duration_s"])
