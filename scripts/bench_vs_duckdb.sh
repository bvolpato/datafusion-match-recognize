#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible apples-to-apples benchmark comparing DuckDB 2.1's native C++ MATCH_RECOGNIZE
# against datafusion-match-recognize sequentially on the same machine over the exact same
# 10,000,000-row Parquet dataset (200,000 partitions x 50 rows).
#
# Usage:
#   ./scripts/bench_vs_duckdb.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${BENCH_THREADS:-8}"
PARTITIONS="${BENCH_PARTITIONS:-200000}" # 200,000 * 50 = 10,000,000 rows
STORAGE="${BENCH_STORAGE:-parquet}"      # parquet (default) or memory

WORK_DIR="${REPO_DIR}/.bench-duckdb-tmp"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/parquet"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "==> Step 1/3: Building DataFusion benchmark binary (--release)"
cargo build --release --manifest-path "${REPO_DIR}/Cargo.toml" --example bench_vs_trino

echo ""
echo "==> Step 2/3: Running DuckDB 2.1 alone (threads=${THREADS}, storage=${STORAGE}, rows=$((PARTITIONS * 50)))"
uv run --prerelease=allow --with 'duckdb>=2.0.1.dev0' python3 - "${PARTITIONS}" "${THREADS}" "${STORAGE}" "${WORK_DIR}" <<'PY'
import json, os, statistics, sys, time, duckdb

num_partitions = int(sys.argv[1])
threads = int(sys.argv[2])
storage = sys.argv[3]
work_dir = sys.argv[4]
parquet_dir = os.path.join(work_dir, "parquet")
checksum_path = os.path.join(work_dir, "checksums.json")
duck_timings_path = os.path.join(work_dir, "duckdb_timings.json")

rows_per_part = 50
warmup_iters = 2
measured_iters = 5

con = duckdb.connect()
con.execute(f"SET threads = {threads}")
print(f"DuckDB version: {duckdb.__version__}, threads={threads}, storage={storage}")

gen_sql = f"""
SELECT
    p::BIGINT AS p,
    seq::BIGINT AS seq,
    (seq * 10)::BIGINT AS ts,
    ((p * 31 + seq * 17) % 10)::BIGINT AS ev,
    (100 + ((p * 7 + seq * 13) % 41) - ((seq * 3) % 19))::BIGINT AS price,
    (CASE
        WHEN p % 100 = 0 AND seq = 5 THEN 0
        WHEN seq = 25 THEN 1
        WHEN seq = 45 THEN 2
        ELSE 9
    END)::BIGINT AS sparse_ev
FROM range(1, {num_partitions + 1}) t1(p)
CROSS JOIN range(1, {rows_per_part + 1}) t2(seq)
"""

if storage == "parquet":
    t0 = time.perf_counter()
    con.execute(f"COPY ({gen_sql}) TO '{parquet_dir}' (FORMAT PARQUET, PER_THREAD_OUTPUT true)")
    print(f"Generated {num_partitions * rows_per_part:,} rows to Parquet in {time.perf_counter() - t0:.2f}s")
    con.execute(f"CREATE VIEW bench_events AS SELECT * FROM read_parquet('{parquet_dir}/*.parquet')")
else:
    t0 = time.perf_counter()
    con.execute(f"CREATE TABLE bench_events AS {gen_sql}")
    print(f"Materialized {num_partitions * rows_per_part:,} rows in memory in {time.perf_counter() - t0:.2f}s")

workloads = [
    (
        "1_funnel_3step_stateless",
        """
        SELECT COUNT(*) AS cnt, COALESCE(SUM(dt), 0) AS total_dt
        FROM bench_events
        MATCH_RECOGNIZE (
            PARTITION BY p
            ORDER BY seq
            MEASURES
                CASE WHEN c.ts IS NOT NULL THEN FIRST(c.ts) - FIRST(a.ts) ELSE 0 END AS dt
            ONE ROW PER MATCH
            AFTER MATCH SKIP PAST LAST ROW
            PATTERN ((a any0* b any1* c) | (a any0* b) | a)
            DEFINE
                a AS ev = 0,
                any0 AS ev <> 1,
                b AS ev = 1,
                any1 AS ev <> 2,
                c AS ev = 2
        )
        """,
    ),
    (
        "2_funnel_sparse_1pct_fast_skip",
        """
        SELECT COUNT(*) AS cnt, COALESCE(SUM(dt), 0) AS total_dt
        FROM bench_events
        MATCH_RECOGNIZE (
            PARTITION BY p
            ORDER BY seq
            MEASURES FIRST(c.ts) - FIRST(a.ts) AS dt
            ONE ROW PER MATCH
            AFTER MATCH SKIP PAST LAST ROW
            PATTERN (a any0* b any1* c)
            DEFINE
                a AS sparse_ev = 0,
                any0 AS sparse_ev <> 1,
                b AS sparse_ev = 1,
                any1 AS sparse_ev <> 2,
                c AS sparse_ev = 2
        )
        """,
    ),
    (
        "3_overlapping_skip_next_all_rows",
        """
        SELECT COUNT(*) AS cnt, COALESCE(SUM(s_price), 0) AS total_s
        FROM bench_events
        MATCH_RECOGNIZE (
            PARTITION BY p
            ORDER BY seq
            MEASURES SUM(price) AS s_price
            ALL ROWS PER MATCH OMIT EMPTY MATCHES
            AFTER MATCH SKIP TO NEXT ROW
            PATTERN (a b+ c)
            DEFINE
                a AS price >= 100,
                b AS price < 115,
                c AS price >= 115
        )
        """,
    ),
    (
        "4_stock_vshape_stateful_prev",
        """
        SELECT COUNT(*) AS cnt, COALESCE(SUM(bottom_p), 0) AS total_bp
        FROM bench_events
        MATCH_RECOGNIZE (
            PARTITION BY p
            ORDER BY seq
            MEASURES LAST(down.price) AS bottom_p
            ONE ROW PER MATCH
            AFTER MATCH SKIP TO LAST up
            PATTERN (strt down+ up+)
            DEFINE
                down AS down.price < PREV(down.price),
                up AS up.price > PREV(up.price)
        )
        """,
    ),
    (
        "5_time_window_funnel_stateful",
        """
        SELECT COUNT(*) AS cnt, COALESCE(SUM(dt), 0) AS total_dt
        FROM bench_events
        MATCH_RECOGNIZE (
            PARTITION BY p
            ORDER BY seq
            MEASURES LAST(b.ts) - FIRST(a.ts) AS dt
            ONE ROW PER MATCH
            AFTER MATCH SKIP PAST LAST ROW
            PATTERN (a any0* b)
            DEFINE
                a AS ev = 0,
                any0 AS ev <> 2,
                b AS ev = 2 AND b.ts - FIRST(a.ts) <= 150
        )
        """,
    ),
]

checksums = {}
timings = {}
for name, sql in workloads:
    for _ in range(warmup_iters):
        con.execute(sql).fetchall()
    runs_ms = []
    out = None
    for _ in range(measured_iters):
        t0 = time.perf_counter()
        out = con.execute(sql).fetchall()
        runs_ms.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(runs_ms)
    formatted = [[str(c) for c in row] for row in out]
    checksums[name] = formatted
    timings[name] = med
    print(f"DUCKDB | `{name}` | wall={med:.1f} ms | out=({', '.join(formatted[0])})")

with open(checksum_path, "w") as f:
    json.dump(checksums, f, indent=2)
with open(duck_timings_path, "w") as f:
    json.dump({"version": duckdb.__version__, "timings_ms": timings}, f, indent=2)
PY

echo ""
echo "==> Step 3/3: Running DataFusion alone (threads=${THREADS}, storage=${STORAGE}, rows=$((PARTITIONS * 50)))"
BENCH_MODE=df \
BENCH_STORAGE="${STORAGE}" \
BENCH_PARTITIONS="${PARTITIONS}" \
BENCH_THREADS="${THREADS}" \
BENCH_PARQUET_DIR="${WORK_DIR}/parquet" \
BENCH_CHECKSUM_FILE="${WORK_DIR}/checksums.json" \
  "${REPO_DIR}/target/release/examples/bench_vs_trino"
