#!/usr/bin/env bash
set -euo pipefail
CLIENT_KEY_FILE="${CLIENT_KEY_FILE:-/home/user/.config/ornith15/api-key}"
key=$(tr -d '\r\n' < "$CLIENT_KEY_FILE")
base="${SPARK_URL:-http://127.0.0.1:30000}"
curl -fsS --max-time 5 "$base/health" | python3 -c 'import json,sys; assert json.load(sys.stdin).get("ok") is True'
bad_status=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
  -H 'Authorization: Bearer wrong-key' "$base/v1/models")
[[ "$bad_status" == 401 ]] || { echo "gateway accepted a wrong key (HTTP $bad_status)" >&2; exit 1; }
curl -fsS --max-time 5 -H "Authorization: Bearer $key" "$base/v1/models" |
  python3 -c 'import json,sys; assert any(m["id"] == "qwen3.8-flash-next" for m in json.load(sys.stdin)["data"])'
curl -fsS --max-time 180 "$base/v1/messages" \
  -H "Authorization: Bearer $key" -H 'anthropic-version: 2023-06-01' \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-flash-next","max_tokens":32,"messages":[{"role":"user","content":"Reply with the word ready."}]}' |
  python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("role")=="assistant" and d.get("content"),d'
curl -fsSN --max-time 180 "$base/v1/messages" \
  -H "Authorization: Bearer $key" -H 'anthropic-version: 2023-06-01' \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-flash-next","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"Reply with the word ready."}]}' |
  python3 -c 'import sys; body=sys.stdin.read(); assert "event: message_start" in body and "event: message_stop" in body,body[:500]'
curl -fsS --max-time 5 "$base/metrics" |
  python3 -c 'import sys; body=sys.stdin.read(); assert "vllm_generation_tokens_total" in body,body[:500]'
printf 'TensorFold gateway, Anthropic response/stream, and dashboard metrics bridge: PASS\n'
