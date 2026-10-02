# Spark port-30000 TensorFold fork

This fork preserves the existing `qwen3.8-flash-next` client alias and the
client-facing `:30000` URL. TensorFold runs privately on `127.0.0.1:8888`.
CLIProxyAPI v8.0.10 translates Claude Code's Anthropic `/v1/messages` requests to
TensorFold's private OpenAI chat API; it listens on `127.0.0.1:30001`,
and nginx retains the existing client
key on `:30000`. No slot launcher or client key needs changing. The existing
Spark dashboard keeps pointing at `http://192.168.68.113:30000` and its current
provider key. Its `/health` and `/v1/models` probes resolve the new model.

The small metrics bridge maps TensorFold's live emitted-token counter and
prompt counter into the names the dashboard's vLLM adapter reads. Never use
completed-response token totals for live throughput: that creates
zeroes during generation and a large fake burst at completion. The running
gauge also comes from TensorFold health, not a gateway's all-HTTP gauge.
Token totals retain the observed count across engine reloads for this bridge
process's lifetime. Restart the dashboard together with a bridge restart to
clear its counter baselines and historical averages. Prompt totals still update
at first output, so the prompt-rate chart is admission accounting, not measured
instantaneous prefill speed (and includes cached prompt tokens).

With `NATIVE_ENGINE_METRICS=1`, the bridge scrapes TensorFold 0.6.1's native
`/metrics` directly, without gateway credentials or a LiteLLM dependency. It maps
the measured model TTFT and engine E2E histograms, queue, KV occupancy, MTP
draft/acceptance totals and preemptions. Engine E2E excludes translation and
network transit. TPOT, ITL and batch remain absent rather than fabricated. This
is dashboard compatibility, **not** vLLM instrumentation. A migration-only
`TOKEN_COUNTER_SEED` JSON file can preserve the previous bridge's observed live
token counters when replacing it; it is not required for a fresh stack.

For a telemetry-only repair, back up the remote bridge file, transfer the new
bridge, run its unit tests on the host, and restart only `spark-tf-metrics` and
`spark-dashboard`. Do not restart the model or gateway and do not move slots.
Older engines without TTFT must have `SPARK_WARMUP_SKIP_REQUESTS=0`; this is
already set on the live dashboard. TensorFold 0.6.1 does export TTFT. Preserve the rest of the
dashboard's container configuration and state volume when setting this flag.
Prove advancing token counters while a response is still streaming and verify
the corresponding WebSocket throughput and E2E values; connectivity alone is
not a chart-correctness proof. Restore the saved bridge and restart those same
two telemetry containers if proof fails.

## Preparation and cutover

### CLIProxyAPI gateway (October 2)

The live route is nginx `:30000` → CLIProxyAPI `127.0.0.1:30001` → TensorFold
`127.0.0.1:8888`. The old `spark-tf-litellm` is stopped and retained for rollback;
it is not serving or loading another model. No Mac CLIProxy instance, slot
launcher, client key, MoP process or conversation was changed.

Prepare the CPU-only gateway image on the Spark with:

```sh
bash spark/prepare-cliproxy.sh
```

This downloads the official Linux/aarch64 no-plugin v8.0.10 archive and checks
SHA256 `fa776f18c4ce486a6d3eaf68ea9d1337bee1865d116b0e3b4d05a5d9c36af2bc`
before building `spark/Dockerfile.cliproxy`. The deployed image ID is
`sha256:822a2ddf3c5f777b368439a82805d6ea05141dfe6bb57da8fa4a5677be71f3b6`.
This local ID may differ on a rebuild; do not substitute an unverified binary.

`spark/cliproxy.yaml.template` uses the existing private gateway key, declares
the upstream text-only, disables management and request logging, and stores no
OAuth credentials. It explicitly disables Claude model-list cloaking, so both
Anthropic and OpenAI model discovery return `qwen3.8-flash-next`. Retries stay
off; streaming keepalive is 15 seconds. The live container is capped at 512 MiB
without extra swap and uses `unless-stopped`.

The built-in text-only guard replaces **tool-returned** images with
`[image omitted: unsupported by upstream]`, preserving adjacent text and tool
IDs. It never turns screenshot base64 into ordinary prompt text. This does not
provide visual understanding: direct image/video prompts remain unsupported.
The live single-image history proof used 349 input tokens despite one million
base64 characters; Anthropic replies, reasoning/text SSE, xhigh tool arguments,
wrong-key rejection and the native dashboard metrics passed.

`start-stack.sh` now creates CLIProxyAPI and the native metrics bridge for a
**fresh, prepared** stack. Do not run it over the live Tinfield services or
preserved rollback containers. For an existing stack, interrupt only active
requests, preserve/stop the exact old gateway, start the prepared CLIProxyAPI
container with the private config mounted read-only, and prove the public
Anthropic route before resuming existing conversations sequentially. If proof
fails, stop only CLIProxyAPI and start the retained LiteLLM gateway. Both bind
the same private port and must never run together. If reverting telemetry too,
stop the new bridge and restore the retained
`spark-tf-metrics-litellm-rollback-20261002` container under its canonical name.

Upstream: [release](https://github.com/router-for-me/CLIProxyAPI/releases/tag/v8.0.10),
[configuration](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.10/config.example.yaml),
[tool-image guard](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.10/internal/runtime/executor/helps/openai_compat_tool_results.go).

### TensorFold 0.6.0 prompt-copy drafting (Victoria NVFP4)

`spark/patches/tf060-prompt-copy.patch` ports Mia's patch 0007 to the 0.6.0
concurrent decoder. It consumes `TENSORFOLD_MTP_COPY=1`, which the unpatched
0.6.0 build ignores. MTP first absorbs the accepted rows; an eight-token
suffix match with a backed continuation then proposes at most the configured
draft depth and remaining reply room. The target verifier, sampling, grammar,
EOS handling, prefix cache, prefill scheduling and model weights are unchanged.
No match falls back to MTP. Mixed copy/MTP rounds select the correct per-stream
logit rows after excluding copied streams. Temporary pending tokens are always
removed from the request context, even if lookup fails.

Each copied request reports `copy_drafted` (copied rows actually verified) and
`copy_accepted` in its existing `tensorfold` stats. Serial requests never use
lookup. `spark/test_prompt_copy.py` runs twelve CPU control-flow tests against
the actual installed sources. `spark/prove_prompt_copy.py` checks four concurrent
greedy/sampled quoting and novel-output requests, baseline token hashes, copied
acceptance and the uncached serial reference. It does not claim a universal
throughput improvement; Mia's older MLX recipe reports a quoting/editing benefit.

Build `spark/Dockerfile.prompt-copy` only after checking its base image is
`sha256:2c318ce3dd7fdee5d1fac9d684382f5ecd444d2c3aefd7484aba7af5ac4f3f4b`.
Pass the patch SHA256 as `PATCH_SHA256`. Replace only the backend in a separately
named container and preserve the stopped source for rollback. Keep the existing
route/auth/alias, 262144 context, four streams, INT4 KV, MTP3/confidence .60 and
decode share .20. Do not replay the older 0.3.x maintenance scripts over this
0.6.0 container. Resume only previously active clients, sequentially, in their
existing conversations after authenticated Anthropic streaming/tool proof.

Source: [Mia's prompt-copy patch](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/main/patches/0007-flash-next-copy-drafts.patch).

### Tinfield-1 EXL3 on TensorFold 0.6.1

`spark/tinfield.py` is a separate, text-only profile for
[khronnuz/Tinfield-1-exl3](https://huggingface.co/khronnuz/Tinfield-1-exl3/tree/4.05bpw_h6_ng6),
the EXL3 conversion of `badtheorylabs/Tinfield-1` (Flash Next, not dense 27B).
The 4.05-bpw / 6-bit-head / 6-bit-ngram branch is pinned to
`460f8565373f20e1c172f72f261f591e4f76b8a4` in `spark/tinfield-exl3.json`.
Do not pull the repo's `main`: that branch has no serving weights.

The profile pins the separately built TensorFold 0.6.1 image
`sha256:21bd28447c82904c0593075f3c29c59eed30500923ea29feb38df62384706c0a`.
This is a **local Spark image ID**, not an image available from a registry.
It uses upstream `v0.6.1` at `17c73e189f5e6a5304cda7ea37f086f9c49b4788`,
retaining the scoped loading estimate and verified prompt-copy changes in
`spark/patches/tf061-tinfield.patch`. No ExLlama runtime is required.
`runtime` checks the installed family,
n-gram loader and copy patch in an offline CPU-only container.

Run these commands **on the Spark** from the recipe checkout:

```sh
python3 spark/test_tinfield.py
python3 spark/tinfield.py runtime
python3 spark/tinfield.py prepare
python3 spark/tinfield.py status
docker logs --tail 30 tinfield-flash-next-tf060-copy-v1-download
python3 spark/tinfield.py check
python3 spark/tinfield.py command
```

`prepare` starts one named Docker download worker, without a GPU,
limited to 512 MiB RAM, no additional swap, two CPUs and one download worker.
Xet and HF Transfer are disabled; the exact revision is downloaded over HTTP.
Restarting reuses completed cached files; HTTP transient retries can resume the
current transfer, but HF 1.24 does not retain byte-resume state across restarts.
The ~100.1 GiB pack includes nine weight shards and a separate ~36.4 GiB n-gram
file, plus its own tokenizer/template and metadata. That extra file is required
even though it is absent from the main weight index. Its 128 six-bit segments
are packed into one file. Do not substitute Victoria's PLE or MTP weights.
`prepare` does not auto-retry an OOM or stop/remove any existing container.
The completed download container and logs remain available for inspection.

`check` rejects partial/wrong-size files, wrong HF blob identities, corrupt
critical LFS metadata, wrong architecture/quantization, missing MTP keys and
invalid table geometry. It then calculates header-only memory estimates using
the actual installed engine in a CPU-only container capped at 2 GiB. It does
not hash every weight byte, load a model, or claim GPU inference is validated.
Its full-four-window estimate is separate from the stock startup estimate:
the engine admits one full window initially and grows other streams subject to
its memory gate. Mapped n-gram pages are reclaimable and may page from SSD if
memory becomes tight. Four configured slots are not proof that four maximum
contexts plus the whole table remain resident. The stock memory guard stays on.

The future launch command keeps private `127.0.0.1:8888`, the public gateway's
existing `:30000` route, `qwen3.8-flash-next` alias, four slots, 262144 context,
INT4 KV, MTP4/confidence .60, thinking, prompt-copy and decode share .20.
Neither `--ple-on-ssd` nor `--prefill-fp8` is valid for EXL3; both are omitted.
Image/video serving remains off. The backend itself does not require gateway
or dashboard changes; the separate October 2 CLIProxyAPI migration above is
already live.

The October 2 draft-depth trial changes only MTP3 to MTP4, using the separately
named `tinfield-flash-next-tf060-copy-mtp4-v1` container. The original
`tinfield-flash-next-tf060-copy-v1` stays stopped as the exact MTP3 rollback.
Matched 72,097-token cached prompts (three repeats, greedy and sampled, one
and two streams) showed only a modest 1.3–2.8% median per-request decode gain;
all output token hashes matched. This is a bounded synthetic result, not a
claim of a large speedup or four simultaneously full 262K contexts. That first
trial did not include a runtime or scheduler change.

The subsequent October 2 update uses the separately named
`tinfield-flash-next-tf061-copy-mtp4-share20-v1` container. Its immediate rollback
is the preserved `tinfield-flash-next-tf060-copy-mtp4-v1` container. Keep both old
containers; do not remove or restart them while the new backend serves.
`decode_share` is explicit in the JSON profile. A same-version .30 trial did not
improve the bounded four-client mixed workload: .20 finished in 35.11 s versus
38.37 s at .30, with identical output hashes; the largest streaming gap was
2.81 s versus 4.55 s. These are single synthetic runs, not production-wide
performance guarantees. Keep .20; the stopped .30 trial remains available for
inspection. Raising the share is intended to give decode more scheduler time
during concurrent prefill, not a 50% increase in standalone tokens/second.
v0.6.1 also introduces
shared kept prefixes and short-prompt-first scheduling on CUDA Flash Next.
The larger idle prompt workspace in this release excludes EXL3, so do not
advertise that path as active for Tinfield.

The later same-image .10 trial also showed no useful gain. In a matched
four-client workload, .20 took 34.30 s and .10 took 35.46 s. The uncached
14,349-token request took 20.52 s and 20.48 s respectively (engine prefill
14.90 s and 14.87 s), with identical output hashes throughout. Maximum
streaming gaps were 2.81 s and 2.84 s. Keep .20: a ~0.2% cold-request difference
is not a meaningful improvement, while overall time was ~3.4% worse. These
are single bounded synthetic runs, not long-context production guarantees.
The .10 container is stopped and retained with restart disabled. The original
.20 container is serving again with its unchanged `unless-stopped` policy;
only the separate CLIProxyAPI/metrics migration was retained.

Rebuild with `spark/Dockerfile.tinfield-tf061` after verifying the base image is
`sha256:2c318ce3dd7fdee5d1fac9d684382f5ecd444d2c3aefd7484aba7af5ac4f3f4b`.
Pass the SHA256 of `spark/patches/tf061-tinfield.patch` as `PATCH_SHA256`, then
replace the profile image ID with the verified local result. The active trial
was built from that exact upstream checkout plus those three changed files,
using the same retained CUDA/PyTorch stack. Slots stay on Sol during validation;
do not return them to DGX without a separate request.

**Preparation is not cutover.** Only after a separately authorized cutover has
moved affected clients off Spark and preserved/stopped the source backend:

```sh
python3 spark/tinfield.py start
```

`start` checks the pack and refuses if port 8888 is occupied or the Tinfield
container already exists. It never stops Victoria, removes a rollback, moves
slots or rebuilds the gateway. The new backend uses `unless-stopped` for boot
restart. Before returning clients, prove actual GPU loading, MTP acceptance,
four concurrent text requests, authenticated Anthropic streaming/tool turns
and advancing dashboard counters. If proof fails, stop only the new backend
and restore the preserved source under the normal cutover procedure. Do not
run the older root `start.sh`, `prepare.sh` or `start-stack.sh` over this stack.

Runtime support: [TensorFold EXL3 recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/exl3.md).

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
