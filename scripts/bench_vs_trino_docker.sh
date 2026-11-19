#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible apples-to-apples benchmark running Trino 484 and datafusion-match-recognize
# sequentially inside identical Linux Docker containers (--cpus=8 --memory=12g) over the exact
# same 10,000,000-row Parquet dataset written by Trino.
#
# Usage:
#   ./scripts/bench_vs_trino_docker.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATAFUSION_DIR="${DATAFUSION_DIR:-$(cd "${REPO_DIR}/../datafusion" && pwd)}"
CPUS="${BENCH_CPUS:-8}"
MEM="${BENCH_MEM:-12g}"
PARTITIONS="${BENCH_PARTITIONS:-200000}" # 200,000 * 50 = 10,000,000 rows
RUST_IMAGE="${BENCH_RUST_IMAGE:-rust:latest}"
TRINO_IMAGE="${BENCH_TRINO_IMAGE:-trinodb/trino:latest}"

WORK_DIR="${REPO_DIR}/.bench-tmp"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/cfg" "${WORK_DIR}/data"
chmod 777 "${WORK_DIR}/data"
trap 'docker rm -f mr-trino-fair >/dev/null 2>&1 || true; rm -rf "${WORK_DIR}"' EXIT

cat > "${WORK_DIR}/cfg/hive.properties" <<'EOF'
connector.name=hive
hive.metastore=file
hive.metastore.catalog.dir=local:///metastore
fs.native-local.enabled=true
local.location=/benchdata
EOF

echo "==> Step 1/2: Running Trino (${TRINO_IMAGE}) alone with --cpus=${CPUS} --memory=${MEM}"
docker rm -f mr-trino-fair >/dev/null 2>&1 || true
docker run -d --name mr-trino-fair --cpus="${CPUS}" --memory="${MEM}" -p 18080:8080 \
  -v "${WORK_DIR}/cfg/hive.properties:/etc/trino/catalog/hive.properties:ro" \
  -v "${WORK_DIR}/data:/benchdata" \
  "${TRINO_IMAGE}" >/dev/null

BENCH_MODE=trino \
BENCH_STORAGE=parquet \
BENCH_PARTITIONS="${PARTITIONS}" \
BENCH_THREADS="${CPUS}" \
BENCH_CHECKSUM_FILE="${WORK_DIR}/data/checksums.json" \
  cargo run --release --manifest-path "${REPO_DIR}/Cargo.toml" --example bench_vs_trino

docker rm -f mr-trino-fair >/dev/null

# Stage Trino's exact generated Parquet data files with .parquet suffix (excluding .trinoSchema).
mkdir -p "${WORK_DIR}/data/parquet"
for f in "${WORK_DIR}/data/metastore/default/bench_events/"*; do
  base="$(basename "$f")"
  [[ "$base" == .* ]] && continue
  mv "$f" "${WORK_DIR}/data/parquet/${base}.parquet"
done
echo "Parquet files written by Trino:"
ls -lh "${WORK_DIR}/data/parquet"

echo ""
echo "==> Step 2/2: Running DataFusion (${RUST_IMAGE}) alone with --cpus=${CPUS} --memory=${MEM}"
docker run --rm --cpus="${CPUS}" --memory="${MEM}" \
  -v "${REPO_DIR}:/workspace/datafusion-match-recognize" \
  -v "${DATAFUSION_DIR}:/workspace/datafusion:ro" \
  -v "${WORK_DIR}/data:/benchdata:ro" \
  -v mr-bench-cargo-cache:/usr/local/cargo/registry \
  -v mr-bench-target-cache:/workspace/target-linux \
  -w /workspace/datafusion-match-recognize \
  -e CARGO_TARGET_DIR=/workspace/target-linux \
  -e BENCH_MODE=df \
  -e BENCH_STORAGE=parquet \
  -e BENCH_PARTITIONS="${PARTITIONS}" \
  -e BENCH_THREADS="${CPUS}" \
  -e BENCH_PARQUET_DIR=/benchdata/parquet \
  -e BENCH_CHECKSUM_FILE=/benchdata/checksums.json \
  "${RUST_IMAGE}" \
  cargo run --release --no-default-features --example bench_vs_trino
