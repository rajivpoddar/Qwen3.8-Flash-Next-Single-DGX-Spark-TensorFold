#!/usr/bin/env bash
# Run only after the old :30000 server has been stopped and every Spark slot has drained.
# This script never stops the source server or changes a slot.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
# shellcheck source=scripts/config.sh
source scripts/config.sh

[[ "$HOST" == 127.0.0.1 && "$PORT" == 8888 && "$SERVED_NAME" == qwen3.8-flash-next ]] ||
  die "this integration requires private TensorFold 127.0.0.1:8888 and alias qwen3.8-flash-next"

CLIENT_KEY_FILE="${CLIENT_KEY_FILE:-/home/user/.config/ornith15/api-key}"
GATEWAY_STATE_DIR="${GATEWAY_STATE_DIR:-$HOME/.config/tensorfold-gateway}"
INTERNAL_KEY_FILE="${INTERNAL_KEY_FILE:-$GATEWAY_STATE_DIR/internal-key}"
CLIPROXY_IMAGE="${CLIPROXY_IMAGE:-spark-tf-cliproxy:v8.0.10}"
NGINX_IMAGE="${NGINX_IMAGE:-nginx:1.27-alpine}"
PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.11-alpine}"

[[ -s "$CLIENT_KEY_FILE" ]] || die "existing client key file missing: $CLIENT_KEY_FILE"
CLIENT_KEY=$(tr -d '\r\n' < "$CLIENT_KEY_FILE")
[[ "$CLIENT_KEY" =~ ^[A-Za-z0-9._~+/-]+$ ]] || die "client key has unsafe nginx configuration characters"
dashboard_env=$(docker inspect spark-dashboard --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null) ||
  die "spark-dashboard container is missing; dashboard integration cannot be proved"
[[ "$(docker inspect -f '{{.State.Running}}' spark-dashboard)" == true ]] ||
  die "spark-dashboard is not running"
dashboard_key=$(printf '%s\n' "$dashboard_env" | sed -n 's/^SPARK_DASHBOARD_ENGINE_API_KEY=//p' | head -1)
dashboard_engine=$(printf '%s\n' "$dashboard_env" | sed -n 's/^SPARK_DASHBOARD_ENGINE=//p' | head -1)
dashboard_url=$(printf '%s\n' "$dashboard_env" | sed -n 's/^SPARK_DASHBOARD_ENGINE_URL=//p' | head -1)
[[ "$dashboard_key" == "$CLIENT_KEY" && "$dashboard_engine" == vllm && "$dashboard_url" == *:30000 ]] ||
  die "dashboard engine, URL or key differs from this gateway; reconcile it before the source is stopped"
[[ -z "$(ss -ltn 'sport = :30000' | sed -n '2p')" ]] || die "port 30000 is still occupied; no source service was stopped"
for name in qwen38-flash-next-tf spark-tf-litellm spark-tf-cliproxy spark-tf-metrics spark-tf-gateway; do
  docker inspect "$name" >/dev/null 2>&1 &&
    die "$name already exists; inspect and remove that exact stopped container before another start"
done
# shellcheck disable=SC2153 # IMAGE comes from scripts/config.sh.
for image in "$IMAGE" "$CLIPROXY_IMAGE" "$NGINX_IMAGE" "$PYTHON_IMAGE"; do
  docker image inspect "$image" >/dev/null 2>&1 || die "image not prepared: $image"
done
[[ "$(prepared_state 2>/dev/null)" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] ||
  die "TensorFold image/checkpoint not prepared; run scripts/prepare.sh while the source still serves"

mkdir -p "$GATEWAY_STATE_DIR"
chmod 700 "$GATEWAY_STATE_DIR"
if [[ ! -s "$INTERNAL_KEY_FILE" ]]; then
  umask 077
  printf 'sk-%s\n' "$(openssl rand -hex 32)" > "$INTERNAL_KEY_FILE"
fi
INTERNAL_KEY=$(tr -d '\r\n' < "$INTERNAL_KEY_FILE")
[[ "$INTERNAL_KEY" == sk-* ]] || die "gateway internal key must start with sk-"
INTERNAL_KEY_JSON=$(jq -Rn --arg key "$INTERNAL_KEY" '$key')
export CLIENT_KEY INTERNAL_KEY INTERNAL_KEY_JSON
umask 077
# shellcheck disable=SC2016 # envsubst receives literal variable names.
envsubst '${CLIENT_KEY} ${INTERNAL_KEY}' < spark/nginx.conf.template > "$GATEWAY_STATE_DIR/nginx.conf"
mkdir -p "$GATEWAY_STATE_DIR/cliproxy/empty-auths"
chmod 700 "$GATEWAY_STATE_DIR/cliproxy" "$GATEWAY_STATE_DIR/cliproxy/empty-auths"
# shellcheck disable=SC2016
envsubst '${INTERNAL_KEY_JSON}' < spark/cliproxy.yaml.template > "$GATEWAY_STATE_DIR/cliproxy/config.yaml"
docker run --rm --network host -v "$GATEWAY_STATE_DIR/nginx.conf:/etc/nginx/nginx.conf:ro" \
  "$NGINX_IMAGE" nginx -t >/dev/null

# The backend starts before the CPU gateway, leaving maximum free RAM for its
# configured KV allocation. PREPARE=0 rules out a surprise download/build now.
PREPARE=0 ./start.sh
docker run -d --name spark-tf-cliproxy --network host --restart unless-stopped \
  --memory=512m --memory-swap=512m \
  -v "$GATEWAY_STATE_DIR/cliproxy:/config:ro" \
  "$CLIPROXY_IMAGE" >/dev/null
docker run -d --name spark-tf-metrics --network host --restart unless-stopped \
  -e NATIVE_ENGINE_METRICS=1 \
  -v "$PWD/spark/metrics_bridge.py:/app/metrics_bridge.py:ro" \
  "$PYTHON_IMAGE" python /app/metrics_bridge.py >/dev/null
docker run -d --name spark-tf-gateway --network host --restart unless-stopped \
  -v "$GATEWAY_STATE_DIR/nginx.conf:/etc/nginx/nginx.conf:ro" \
  "$NGINX_IMAGE" >/dev/null

for ((i=0; i<30; i++)); do
  if curl -fsS --max-time 5 -H "Authorization: Bearer $CLIENT_KEY" \
    http://127.0.0.1:30000/v1/models | \
    python3 -c 'import json,sys; assert any(x["id"] == "qwen3.8-flash-next" for x in json.load(sys.stdin)["data"])' 2>/dev/null; then
    docker update --restart unless-stopped "$CONTAINER_NAME" >/dev/null
    log "gateway serves qwen3.8-flash-next at :30000; run spark/prove-stack.sh before returning slots"
    exit 0
  fi
  sleep 2
done
die "gateway failed model discovery; inspect exact target containers, then restore the source"
