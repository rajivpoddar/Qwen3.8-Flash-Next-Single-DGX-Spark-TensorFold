# Shared settings for start.sh, stop.sh and scripts/*.sh. Any value can be overridden from the environment,
# e.g. `PORT=9000 ./start.sh` or `PULL=0 scripts/prepare.sh`.

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"   # MLX 4-bit, group size 32, with the MTP head
# The patches and start.sh's flags are made for TensorFold v0.3.6.3 exactly (earlier releases lack --vision for
# this model and the patches do not apply). After changing TF_VERSION, TF_REPO or BASE_IMAGE, run
# `scripts/prepare.sh --rebuild`.
TF_VERSION="${TF_VERSION:-v0.3.6.3}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_VERSION}-prefix-retention}"   # local fork: patches 0001-0011
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"          # the server's container
# The prebuilt image: prepare.sh pulls $GHCR_IMAGE:<TF_VERSION>-<patches hash>; publish-image.sh pushes it.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"

SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"   # preserve the existing Spark client model alias
HOST="${HOST:-127.0.0.1}"                           # TensorFold is private behind the Anthropic-compatible gateway
PORT="${PORT:-8888}"                                 # internal TensorFold port; clients stay on public port 30000
# Serving defaults (./start.sh arguments come after them and win). All streams share one memory pool (~103-104 GiB
# budget on a 128 GB Spark, 75 GiB of it weights), so window x streams x KV bytes must fit: 4 streams x 262,144 tokens
# at int8 KV is ~97.8 GiB, 5 streams ~102.6 GiB (~4.5 GiB a stream). Other fits: 3 streams bf16 at 262k, 6 streams
# int4 at 262k, 8 streams int4 at ~250k (tight), 6 streams int8 at ~220k. int4 and bf16 KV change the output slightly.
PARALLEL="${PARALLEL:-4}"          # approved Spark deployment: four concurrent streams
CONTEXT="${CONTEXT:-262144}"       # prompt + reply window per stream (the model's native maximum)
KV_DTYPE="${KV_DTYPE:-int4}"       # approved memory fit; preserves the full 262K context per stream
PLE_ON_SSD="${PLE_ON_SSD:-1}"      # 1: read the 29.8 GiB n-gram tables from SSD, leaving that RAM to the KV cache
# Image and video input (patch 0008): the model's own vision tower, 0.84 GiB. Its ~0.8 GiB of scratch is taken only
# while an image or video encodes and handed back right after, so startup reserves none for it; 2,048-row prompt
# chunks (instead of 4,096) make room for the tower, keeping the full 5 x 262,144 KV pool. VISION=0: text only
# (and 4,096-row chunks, ~2-5% faster prefill from 3k tokens).
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"    # 1: also accept public https:// image and video URLs (default: data URLs only)
# MTP drafting: at most MTP_DRAFTS drafts a round, a chain stopping before a draft under MTP_CONFIDENCE.
# Swept 2026-09-29 (identical output in every arm): 6/0.60 beat the stock 6/0.30 by ~3% on
# prose and ~4% on code, the best balance of both; 4/0.50, 3/0.30 and 7/0.75 matched it on prose but not on code.
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
# Thinking mode (Qwen's recommended sampling): temperature 1.0, top_p 0.95, top_k 20. A request's own values win.
# min_p 0.0, presence_penalty 0.0 and repetition_penalty 1.0 are what TensorFold always does (it has no such
# settings: those values mean "off"). THINKING=0 serves without a think block by default; a request can still set
# "chat_template_kwargs": {"enable_thinking": true|false}.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
THINKING="${THINKING:-1}"
# TensorFold switches (start.sh passes every TENSORFOLD_* variable into the container).
# Prompt chunk rows (patch 0006): 4,096 is +2-5% prefill from 3k tokens over 2,048, and +0.94 GiB at startup, the
# room the vision tower takes: 2,048 with VISION=1, 4,096 without.
if [[ "$VISION" == 1 ]]; then _rows=2048; else _rows=4096; fi
export TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-$_rows}"
# Decode share between committed prefill chunks (patch 0010). 0 restores one round per chunk.
# .20 reserves ~20% of each chunk/decode cycle, capped at .5s extra decode per chunk.
export TENSORFOLD_PREFILL_DECODE_SHARE="${TENSORFOLD_PREFILL_DECODE_SHARE:-0.20}"
# What startup reserves for the vision tower's scratch (MiB); 0: it comes from the system reserve while it encodes.
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"
# The whole video's token budget (Qwen3-VL's per-frame sizing; 2 frames a second, at most 256 frames).
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
# Prompt-lookup drafts ahead of MTP (patch 0007; with PARALLEL >= 2): +6% on replies that repeat the prompt, prose and
# code unchanged. 0: off.
export TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"
# No "is there a newer TensorFold" call to GitHub at each start: the patches are for v0.3.6.3 anyway. 0: check.
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Persists compiled CUDA kernels (torch extensions + triton) so only the first start pays the compile.
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"

MIN_FREE_GB="${MIN_FREE_GB:-125}"   # free disk the checkpoint download needs (it is ~114 GB)
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"   # free disk under Docker's root that pulling or building the image needs

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }

# What scripts/prepare.sh last left ready (it writes this line to PREPARED_MARKER when it succeeds); start.sh runs
# prepare.sh again whenever the current line differs: a missing or stale image, new patches, another model.
PREPARED_MARKER="$KERNEL_CACHE/.prepared"
prepared_state() {
  local hash label model=missing
  hash=$(cat patches/*.patch 2>/dev/null | sha256sum | cut -c1-12)
  label=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || echo missing)
  ls -d "$(model_cache_dir)"/snapshots/*/ >/dev/null 2>&1 && model=present
  echo "model=$MODEL_ID($model) image=$IMAGE($label) patches=$hash"
}
