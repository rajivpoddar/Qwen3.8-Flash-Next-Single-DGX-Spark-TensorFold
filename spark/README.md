# Spark port-30000 TensorFold fork

This fork preserves the existing `qwen3.8-flash-next` client alias and the
client-facing `:30000` URL. TensorFold runs privately on `127.0.0.1:8888`.
LiteLLM translates Claude Code's Anthropic `/v1/messages` requests to
TensorFold's private OpenAI chat API; LiteLLM listens on `127.0.0.1:30001`,
and nginx retains the existing client
key on `:30000`. No slot launcher or client key needs changing. The existing
Spark dashboard keeps pointing at `http://192.168.68.113:30000` and its current
provider key. Its `/health` and `/v1/models` probes resolve the new model.

The small metrics bridge maps TensorFold's live emitted-token counter and
prompt counter into the names the dashboard's vLLM adapter reads. Never use
LiteLLM's completed-response token totals for live throughput: that creates
zeroes during generation and a large fake burst at completion. The running
gauge also comes from TensorFold health, not LiteLLM's all-HTTP gauge.
Token totals retain the observed count across engine reloads for this bridge
process's lifetime. Restart the dashboard together with a bridge restart to
clear its counter baselines and historical averages. Prompt totals still update
at first output, so the prompt-rate chart is admission accounting, not measured
instantaneous prefill speed (and includes cached prompt tokens).

Gateway end-to-end latency sum/count and histogram buckets are mapped to E2E;
they measure complete gateway requests, including prefill and streaming.
Gateway first-frame latency is not mapped to TTFT because the role frame arrives
before prefill. True model TTFT, TPOT, ITL, batch, KV, queue and speculation
metrics remain unavailable until the engine instruments them; do not interpret
them as zero. This is dashboard compatibility, **not** vLLM instrumentation.

For a telemetry-only repair, back up the remote bridge file, transfer the new
bridge, run its unit tests on the host, and restart only `spark-tf-metrics` and
`spark-dashboard`. Do not restart the model or gateway and do not move slots.
The dashboard must have `SPARK_WARMUP_SKIP_REQUESTS=0` for this partial-metrics
backend: its default warmup gate otherwise waits forever for a model TTFT
histogram count that TensorFold does not export. Preserve the rest of the
dashboard's container configuration and state volume when setting this flag.
Prove advancing token counters while a response is still streaming and verify
the corresponding WebSocket throughput and E2E values; connectivity alone is
not a chart-correctness proof. Restore the saved bridge and restart those same
two telemetry containers if proof fails.

## Preparation and cutover

### Retaining a reusable prefix after warm requests

Patch `0011` keeps the original nonfinal snapshot when a resumed prompt has
only a final tail chunk. Previously that request replaced the reusable
checkpoint with the whole prompt; strict-prefix selection made the following
identical request cold again. A new nonfinal chunk still advances the checkpoint.
The existing snapshot and MTP tail are reused without another KV pool or model.
Cold short prompts, images and no-draft requests retain their original behavior.

Build `spark/Dockerfile.prefix-retention` from the preserved decode-budget image,
with the hash of all recipe patches. The installed-source CPU suite has 23
tests, including repeated admission, checkpoint advancement and the MTP successor
boundary. The maintenance harness first compares greedy and seeded sampled warm
output with uncached references, then sweeps 3/4/6 drafts at confidence .30/.60
on four concurrent 65K-token prompts. All four full 262K INT4 buffers and vision
remain allocated. It recommends a change only after alternating confirmation
shows at least 5% combined gain with no greater than 5% workload regression.

The dated `run_prefix_retention_trial.py` is pinned to the captured source/image,
retains the exact rollback container and automatically restores it on failure.
Do not replay it over another serving tuple. API cache, authenticated Anthropic
streaming and tool-turn parity must pass before previously active slots resume
sequentially in their existing Claude sessions. Gateway/dashboard/MoP remain
untouched; `unless-stopped` is retained for the replacement backend.

### Bounded decode share during prefill

Patch `0010` extends the committed-chunk yield to multiple decode rounds.
`TENSORFOLD_PREFILL_DECODE_SHARE=0.20` gives decode a budget equal to one quarter
of the preceding completed CUDA chunk, approximately 20% of the combined cycle.
The burst stops after its time budget, at most 0.5 seconds (plus one indivisible
round), or 32 rounds, whichever is reached first. It returns early when the
existing streams finish; no queued prompt is admitted inside the callback.
Zero retains the prior one-round minimum. No extra model, KV state or scratch
buffer is allocated. Standalone synchronous prefill/warmup is unchanged.

Build `spark/Dockerfile.decode-budget` from the preserved prefill-yield image.
Its 17 CPU tests exercise the installed scheduler and chunk-loop source. A
compatible reload keeps Claude processes/conversations open: interrupt only
affected request turns, retain the stopped old container, run the same cold
65K mixed-load proof before/after, and require identical token hashes. Prove
authenticated discovery and Anthropic streaming before sequentially resuming
previously active clients. Keep the gateway, dashboard and MoP unchanged.

### Reloading the cooperative-prefill patch

Patch `0009` decodes existing streams between committed prefill chunks. It never
recursively admits another prompt or shares the prefill scratch buffer with
decode. For long text prompts it retains one checkpoint at the last full-chunk
boundary instead of the chat template's final token, which can retokenize when
a thinking/tool turn is appended. The MTP successor row is excluded from that
checkpoint so the resumed request recomputes it correctly. This uses the same
four KV states, not another model or KV pool; image inputs retain their original
no-prefix-reuse behavior.

From the captured `tensorfold-qwen38:v0.3.6.3` image, build the bounded derived
image with `spark/Dockerfile.prefill-yield`, setting `PATCHES_HASH` to the hash
of all recipe patches. Its twelve CPU tests exercise the installed source.
Move Spark clients to Muse with `--continue` before stopping the source. Rename
the stopped source container to preserve exact rollback, then start the patched
image with `PREPARE=0 ./start.sh`. Keep the existing gateway, metrics bridge and
dashboard running; do not rerun `start-stack.sh` over these containers.

Before returning clients, run `spark/prove_prefill_fairness.py --candidate`,
`spark/prove_tool_prefix_cache.py`, the Anthropic gateway proofs and dashboard
proof. Require advancing decode during cold long prefill, cache-hit output
identity against uncached serial output, positive MTP acceptance and no OOM or
unexpected restart. Return clients one at a time with `--continue`.

### Original vLLM-to-TensorFold preparation

The validated default is four streams, each with a full 262144-token window,
INT4 KV, and PLE SSD offload. On 2026-09-30, startup admitted 91.17 GiB against
a 95.92 GiB budget with TensorFold's guard intact. This required temporary host
settings `vm.min_free_kbytes=2097152` and `vm.watermark_scale_factor=10`.
The existing persisted vLLM profile remains 4194304/300: reboot restores it and
can block TensorFold admission again. Do not assume Docker's `unless-stopped`
policy alone provides reboot readiness; persist the approved VM profile first.

The host cannot hold the current vLLM model and TensorFold together. First,
while vLLM still serves, prepare the TensorFold image/checkpoint with upstream
`scripts/prepare.sh` and pre-pull ARM64 images
`ghcr.io/berriai/litellm:v1.93.0`, `nginx:1.27-alpine`, and
`python:3.11-alpine`. Build the pinned gateway compatibility image with
`docker build -t spark-tf-litellm:v1.93.0-reasoning-split -f spark/Dockerfile.litellm .`.
Its five focused tests verify that combined reasoning/answer chunks preserve
both fields through the Anthropic streaming adapter. The `hosted_vllm` provider
selects chat-completions translation rather than the unsupported Responses API.
The exact source container is currently `vllm-fn-tp1`; verify it again before
acting. Capture its image, run arguments, env/restart
policy and dashboard configuration for rollback. Do not run `start-stack.sh`
until all Spark-backed slots have moved off this backend, the source has been
stopped, and port 30000 is free. The script never stops the source or slots.

Run `spark/start-stack.sh`, then `spark/prove-stack.sh` on the Spark and
`python3 spark/prove_dashboard.py` from the Mac (which has `websockets`). Check
a real Claude Code slot with streaming and a tool call before returning
remaining slots one at a time. The dashboard should show the alias and measured
prompt/decode counters; the WebSocket proof checks live engine state, not only
HTTP 200.

If proof fails, stop exact target containers `spark-tf-gateway`,
`spark-tf-metrics`, `spark-tf-litellm`, and `qwen38-flash-next-tf`, then restore
the captured source container with its original image/arguments and verify
`:30000/v1/models`, an Anthropic reply, and the dashboard. Keep slots off Spark
until the source is healthy. `spark/start-stack.sh` does not automate the
source stop or rollback; this prevents a blind restart over active work.
