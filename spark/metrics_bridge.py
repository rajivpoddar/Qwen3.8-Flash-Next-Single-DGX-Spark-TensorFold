"""Expose live TensorFold tokens and measured gateway E2E latency to the dashboard.

Never use completed-request token totals as live throughput, or the gateway's
early role frame as model TTFT. Uninstrumented engine metrics stay absent.
"""

import json
import math
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)")
LABEL = re.compile(r'(?:^|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
MAPPED = {
    "litellm_request_total_latency_metric_sum": "vllm_e2e_request_latency_seconds_sum",
    "litellm_request_total_latency_metric_count": "vllm_e2e_request_latency_seconds_count",
    "litellm_request_total_latency_metric_bucket": "vllm_e2e_request_latency_seconds_bucket",
}
TOKEN_FIELDS = {
    "prompt_tokens_total": "vllm_prompt_tokens_total",
    "completion_tokens_total": "vllm_generation_tokens_total",
}


def nonnegative(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid TensorFold metric")
    if value < 0 or not math.isfinite(value) or (integer and type(value) is not int):
        raise ValueError("invalid TensorFold metric")
    return value


class LiveCounters:
    """Keep engine counters monotonic over model reloads during this bridge run.

    The dashboard subtracts adjacent samples without a reset guard. Retain the
    last observed total when a backend resets, then add the new process's tokens.
    Scrapes are serialized by Handler so an older response cannot look like a reset.
    """

    def __init__(self):
        self.previous = {}
        self.totals = {}

    def observe(self, health):
        values = {key: nonnegative(health[key], integer=True) for key in TOKEN_FIELDS}
        nonnegative(health["requests_running"], integer=True)
        result = dict(health)
        for key, current in values.items():
            previous = self.previous.get(key)
            delta = current if previous is None or current < previous else current - previous
            self.totals[key] = self.totals.get(key, 0) + delta
            self.previous[key] = current
            result[key] = self.totals[key]
        return result


def translate(raw: str, health: dict) -> str:
    totals = {name: nonnegative(health[key], integer=True) for key, name in TOKEN_FIELDS.items()}
    totals["vllm_num_requests_running"] = nonnegative(health["requests_running"], integer=True)
    buckets = {}
    for line in raw.splitlines():
        match = SAMPLE.match(line)
        if not match or match[1] not in MAPPED:
            continue
        try:
            value = float(match[3])
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            labels = dict(LABEL.findall(match[2] or ""))
            if labels.get("model", "qwen3.8-flash-next") != "qwen3.8-flash-next":
                continue
            name = MAPPED[match[1]]
            if name.endswith("_bucket"):
                try:
                    upper = float(labels["le"])
                except (KeyError, ValueError):
                    continue
                if math.isnan(upper) or upper < 0:
                    continue
                buckets[upper] = buckets.get(upper, 0.0) + value
            else:
                totals[name] = totals.get(name, 0.0) + value
    # Do not map gateway TTFT: TensorFold sends the assistant-role frame before
    # prefill, so first gateway frame latency is not model first-token latency.
    # LiteLLM's HTTP in-flight gauge includes the metrics scrape itself. Use
    # TensorFold's actual inference gauge instead of reporting a phantom job.
    lines = ["# TYPE vllm_generation_tokens_total counter\n",
             "# TYPE vllm_prompt_tokens_total counter\n",
             "# TYPE vllm_num_requests_running gauge\n"]
    if "vllm_e2e_request_latency_seconds_count" in totals:
        lines.append("# TYPE vllm_e2e_request_latency_seconds histogram\n")
    for upper, value in sorted(buckets.items()):
        bound = "+Inf" if math.isinf(upper) else format(upper, ".17g")
        lines.append(f'vllm_e2e_request_latency_seconds_bucket{{le="{bound}"}} {value:.17g}\n')
    lines.extend(f"{name} {value:.17g}\n" for name, value in sorted(totals.items()))
    return "".join(lines)


class Handler(BaseHTTPRequestHandler):
    scrape_lock = threading.Lock()
    counters = LiveCounters()

    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        request = Request(
            "http://127.0.0.1:30001/metrics",
            headers={"Authorization": "Bearer " + os.environ["LITELLM_MASTER_KEY"]},
        )
        try:
            with self.scrape_lock:
                with urlopen('http://127.0.0.1:8888/health', timeout=0.8) as response:
                    health = self.counters.observe(json.load(response))
                # A gateway metrics fault must not erase live engine throughput.
                raw = ""
                try:
                    with urlopen(request, timeout=0.8) as response:
                        raw = response.read().decode("utf-8", "replace")
                except OSError as exc:
                    print(f"gateway latency scrape failed: {type(exc).__name__}", flush=True)
                body = translate(raw, health).encode()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.send_error(502, "TensorFold metrics unavailable")
            print(f"metrics scrape failed: {type(exc).__name__}", flush=True)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    if not os.environ.get("LITELLM_MASTER_KEY", "").startswith("sk-"):
        raise SystemExit("LITELLM_MASTER_KEY must start with sk-")
    ThreadingHTTPServer(("127.0.0.1", 30002), Handler).serve_forever()
