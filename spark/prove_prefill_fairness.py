"""Synthetic same-port proof of decode fairness during a cold 65K prefill.

Run --baseline on the preserved original service, then --candidate after reload.
No tools are executed and no user material or API credentials are involved.
"""
import concurrent.futures
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
CANDIDATE = "--candidate" in sys.argv
events = []
started = threading.Event()


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def stream_short():
    body = {"model": "qwen3.8-flash-next", "max_tokens": 768, "temperature": 0,
            "stream": True, "ignore_eos": True, "draft": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Write an extensive Python implementation of a red-black tree, with insertion, deletion and invariant checks. Output code only."}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text, final = "", None
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            item = json.loads(line[6:])
            for choice in item.get("choices", []):
                part = choice.get("delta", {}).get("content", "")
                if part:
                    text += part
                    events.append(time.monotonic())
                    started.set()
            if item.get("tensorfold"):
                final = item
    assert text, "no decoded text"
    assert final is not None, "missing final decoder statistics"
    return final


assert call("/health")["requests_running"] == 0, "proof requires no slot traffic"
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    short = pool.submit(stream_short)
    assert started.wait(30), "short request did not start decoding"
    time.sleep(0.15)
    long_started = time.monotonic()
    # Unique filler: must be a cold admission, not an extension of a prior cached prompt.
    long_body = {"model": "qwen3.8-flash-next", "max_tokens": 32, "temperature": 0,
                 "chat_template_kwargs": {"enable_thinking": False},
                 "messages": [{"role": "user", "content": f"Cold probe {time.time_ns()}. Ignore the filler, then reply LONG_PREFILL_READY.\n" +
                               "alpha beta gamma delta epsilon zeta eta theta\n" * 6500}]}
    long = pool.submit(call, "/v1/chat/completions", long_body)
    long_result = long.result(timeout=180)
    long_finished = time.monotonic()
    short_result = short.result(timeout=180)
assert "LONG_PREFILL_READY" in long_result["choices"][0]["message"]["content"]
assert long_result["tensorfold"]["cached"] == 0, "long prompt was not cold"
gaps = [b - a for a, b in zip(events, events[1:])
        if b >= long_started and a <= long_finished]
assert gaps, "no overlap between decode and the new request"
prefill_end = long_started + long_result["tensorfold"]["prefill_s"]
during = [t for t in events if long_started + 1 < t < prefill_end - 1]
short_tokens = 1 + short_result["tensorfold"]["rounds"] + short_result["tensorfold"]["accepted"]
assert short_tokens == 768, "short request did not complete its exact token budget"
result = {"mode": "candidate" if CANDIDATE else "baseline",
          "long_prompt_tokens": long_result["usage"]["prompt_tokens"],
          "long_prefill_s": long_result["tensorfold"]["prefill_s"],
          "decode_text_chunks_during_prefill": len(during),
          "max_decode_gap_during_long_request_s": round(max(gaps), 3),
          "short_stats": short_result.get("tensorfold") if short_result else None,
          "short_decode_tok_s": short_tokens / short_result["tensorfold"]["decode_s"],
          "short_completion_tokens": short_tokens,
          "long_wall_s": long_finished - long_started,
          "final_health": call("/health")}
print(json.dumps(result), flush=True)
assert result["final_health"]["requests_running"] == 0
if CANDIDATE:
    assert len(during) >= 5, "existing decode still blocked behind prefill"
    assert max(gaps) < 5, "decode gap exceeds five-second fairness bound"
    assert result["short_stats"]["accepted"] > 0, "no accepted speculative draft"
    if "--expected-sha" in sys.argv:
        expected = sys.argv[sys.argv.index("--expected-sha") + 1]
        assert result["short_stats"]["token_sha"] == expected, "mixed decode differs from source output"
