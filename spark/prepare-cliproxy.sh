#!/usr/bin/env bash
# CPU-only gateway preparation; never touches live containers or credentials.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
[[ "$(uname -m)" == aarch64 ]] || { echo 'Run on the aarch64 Spark.' >&2; exit 1; }
cliproxy_build_dir=$(mktemp -d /tmp/spark-cliproxy-build.XXXXXXXX)
archive="$cliproxy_build_dir/release.tar.gz"
curl --fail --location --retry 3 --output "$archive" \
  https://github.com/router-for-me/CLIProxyAPI/releases/download/v8.0.10/CLIProxyAPI_8.0.10_linux_aarch64_no-plugin.tar.gz
printf '%s  %s\n' fa776f18c4ce486a6d3eaf68ea9d1337bee1865d116b0e3b4d05a5d9c36af2bc "$archive" | sha256sum --check
tar --extract --gzip --no-same-owner --file "$archive" --directory "$cliproxy_build_dir" cli-proxy-api
install -m 0644 spark/Dockerfile.cliproxy "$cliproxy_build_dir/Dockerfile"
docker build --tag spark-tf-cliproxy:v8.0.10 "$cliproxy_build_dir"
printf 'Gateway image prepared; retained build artifacts: %s\n' "$cliproxy_build_dir"
