#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs DuckDB's own `benchmark/micro/match_recognize/*.benchmark` suite (13 self-contained
# SQL:2016 benchmarks covering 200k to 10M rows) sequentially on DuckDB 2.1 and
# datafusion-match-recognize over identical Parquet files, verifying exact checksum parity.
#
# Usage:
#   ./scripts/bench_duckdb_suite.sh
#   BENCH_FILTER=bin_packing ./scripts/bench_duckdb_suite.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${BENCH_THREADS:-8}"
FILTER="${BENCH_FILTER:-}"
WORK_DIR="${BENCH_WORK_DIR:-${REPO_DIR}/.bench-duckdb-suite-tmp}"
KEEP_DIR="${BENCH_KEEP_DIR:-0}"

if [[ "${KEEP_DIR}" != "1" ]]; then
  rm -rf "${WORK_DIR}"
  trap 'rm -rf "${WORK_DIR}"' EXIT
fi
mkdir -p "${WORK_DIR}"

echo "==> Step 1/3: Building DataFusion DuckDB-suite binary (--release)"
cargo build --release --manifest-path "${REPO_DIR}/Cargo.toml" --example bench_duckdb_suite

echo ""
echo "==> Step 2/3: Generating Parquet tables and running DuckDB 2.1 alone (threads=${THREADS})"
uv run --prerelease=allow --with 'duckdb>=2.0.1.dev0' python3 - "${THREADS}" "${WORK_DIR}" "${FILTER}" <<'PY'
import json, os, statistics, sys, time, duckdb

threads = int(sys.argv[1])
work_dir = sys.argv[2]
flt = sys.argv[3]
checksum_path = os.path.join(work_dir, "checksums.json")

con = duckdb.connect()
con.execute(f"SET threads = {threads}")
print(f"DuckDB version: {duckdb.__version__}, threads={threads}")

tables = {
    "hits_sessionize": """
        SELECT (i // 1000)::INTEGER AS visitor, i::INTEGER AS seq,
               (i * 10 + (i // 50) * 1000)::BIGINT AS ts
        FROM range(0, 10000000) t(i)
    """,
    "readings_streak": """
        SELECT (i // 1000)::INTEGER AS sensor, i::INTEGER AS i, (hash(i) % 1000)::INTEGER AS v
        FROM range(0, 10000000) t(i)
    """,
    "hits_funnel": """
        SELECT (i // 10)::INTEGER AS visitor, i::INTEGER AS seq,
               (['browse', 'select', 'checkout', 'idle'])[(i % 4) + 1] AS action
        FROM range(0, 10000000) t(i)
    """,
    "ticks_sel": """
        SELECT i::INTEGER AS i, (hash(i) % 100)::INTEGER AS v
        FROM range(0, 10000000) t(i)
    """,
    "readings_agg": """
        SELECT i::INTEGER AS i, (i::BIGINT * 2654435761 % 1000003 % 100)::INTEGER AS v
        FROM range(0, 1000000) t(i)
    """,
    "items_bin": """
        SELECT i::BIGINT AS i, (i // 1000)::BIGINT AS bin,
               1 + ((i::BIGINT * 2654435761 % 1000003) % 9)::INTEGER AS w
        FROM range(0, 1000000) t(i)
    """,
    "spans_merge": """
        SELECT s::BIGINT AS s, (s + 1 + (h % 4))::BIGINT AS e
        FROM (
            SELECT i * 2 + (h % 3)::BIGINT AS s, h
            FROM (SELECT i, (i::BIGINT * 2654435761 % 1000003) AS h FROM range(0, 1000000) t(i))
        )
    """,
    "ticks_shoulders": """
        SELECT i::BIGINT AS i,
               (100 + ((i // 25) % 7) * 2 - abs((i % 25) - 12) * (1 + ((i // 25) % 3)) + (h % 3))::INTEGER AS v
        FROM (SELECT i, (i::BIGINT * 2654435761 % 1000003) AS h FROM range(0, 1000000) t(i))
    """,
    "readings_nav": """
        SELECT i::INTEGER AS i, (i::BIGINT * 2654435761 % 1000)::INTEGER AS v
        FROM range(0, 200000) t(i)
    """,
    "readings_quant": """
        SELECT i::INTEGER AS i, (i % 17)::INTEGER AS v
        FROM range(0, 500000) t(i)
    """,
    "runs_no_match": """
        SELECT i::INTEGER AS seq,
               CASE WHEN i % 3000 < 1000 THEN 'browse'
                    WHEN i % 3000 < 2000 THEN 'select'
                    ELSE 'checkout' END AS action
        FROM range(0, 200000) t(i)
    """,
    "run_nested": """
        SELECT i::INTEGER AS i, 'a' AS c FROM range(0, 2000000) t(i)
    """,
}

suite = [
    (
        "sessionize_10m",
        "hits_sessionize",
        """
        SELECT count(*), max(len) FROM hits_sessionize MATCH_RECOGNIZE(
            PARTITION BY visitor ORDER BY seq
            MEASURES count(*) AS len
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN (A B*)
            DEFINE B AS B.ts - PREV(B.ts) <= 100
        )
        """,
    ),
    (
        "streak_10m",
        "readings_streak",
        """
        SELECT max(len) FROM readings_streak MATCH_RECOGNIZE(
            PARTITION BY sensor ORDER BY i
            MEASURES count(*) AS len
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN (A B*)
            DEFINE B AS B.v > PREV(B.v)
        )
        """,
    ),
    (
        "funnel_three_step_10m",
        "hits_funnel",
        """
        SELECT count(*) FROM hits_funnel MATCH_RECOGNIZE(
            PARTITION BY visitor ORDER BY seq
            MEASURES count(*) AS steps
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN (BROWSE SELECT_IT CHECKOUT)
            DEFINE BROWSE AS BROWSE.action = 'browse', SELECT_IT AS SELECT_IT.action = 'select',
                   CHECKOUT AS CHECKOUT.action = 'checkout'
        )
        """,
    ),
    (
        "funnel_two_step_10m",
        "hits_funnel",
        """
        SELECT count(*) FROM hits_funnel MATCH_RECOGNIZE(
            PARTITION BY visitor ORDER BY seq
            MEASURES count(*) AS steps
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN ((BROWSE SELECT_IT) | BROWSE)
            DEFINE BROWSE AS BROWSE.action = 'browse', SELECT_IT AS SELECT_IT.action = 'select'
        )
        """,
    ),
    (
        "selectivity_mid_10m",
        "ticks_sel",
        """
        SELECT count(*) FROM ticks_sel MATCH_RECOGNIZE(
            ORDER BY i
            MEASURES count(*) AS len
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN (A B{2})
            DEFINE A AS A.v < 25, B AS B.v < 50
        )
        """,
    ),
    (
        "aggregate_define_1m",
        "readings_agg",
        """
        SELECT count(*) FROM readings_agg MATCH_RECOGNIZE(
            ORDER BY i
            MEASURES count(*) AS len
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN (A+)
            DEFINE A AS A.v >= avg(A.v)
        )
        """,
    ),
    (
        "bin_packing_1m",
        "items_bin",
        """
        SELECT count(*), sum(total), sum(packed) FROM items_bin MATCH_RECOGNIZE(
            PARTITION BY bin ORDER BY i
            MEASURES sum(X.w) AS total, count(X.i) AS packed
            ONE ROW PER MATCH
            PATTERN ((X | ({- REST -}))+)
            DEFINE X AS sum(X.w) <= 20
        )
        """,
    ),
    (
        "merge_ranges_1m",
        "spans_merge",
        """
        SELECT count(*), sum(n), max(e - s) FROM spans_merge MATCH_RECOGNIZE(
            ORDER BY s, e
            MEASURES first(s) AS s, max(e) AS e, count(*) AS n
            ONE ROW PER MATCH
            PATTERN (J* L)
            DEFINE J AS NEXT(s) <= max(e)
        )
        """,
    ),
    (
        "shoulders_1m",
        "ticks_shoulders",
        """
        SELECT count(*), sum(len) FROM ticks_shoulders MATCH_RECOGNIZE(
            ORDER BY i
            MEASURES last(i) - first(i) + 1 AS len
            ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
            PATTERN (S UP+ DOWN+ UP2+ DOWN+ UP+ DOWN2+)
            DEFINE UP AS v > PREV(v), DOWN AS v < PREV(v), UP2 AS v > max(UP.v), DOWN2 AS v < min(DOWN.v)
        )
        """,
    ),
    (
        "navigation_200k",
        "readings_nav",
        """
        SELECT count(*) FROM readings_nav MATCH_RECOGNIZE(
            ORDER BY i
            MEASURES i AS start_i
            ONE ROW PER MATCH
            PATTERN (A B+)
            DEFINE A AS true, B AS v > LAST(A.v)
        )
        """,
    ),
    (
        "quantifier_500k",
        "readings_quant",
        """
        SELECT count(*) FROM readings_quant MATCH_RECOGNIZE(
            ORDER BY i
            MEASURES i AS start_i
            ALL ROWS PER MATCH
            PATTERN (A+ B+ A+)
            DEFINE A AS v < 12, B AS v >= 12
        )
        """,
    ),
    (
        "no_match_200k",
        "runs_no_match",
        """
        SELECT count(*) FROM runs_no_match MATCH_RECOGNIZE(
            ORDER BY seq
            MEASURES count(*) AS steps
            ONE ROW PER MATCH
            PATTERN (BROWSE+ SELECT_IT+ CHECKOUT+ REFUND)
            DEFINE BROWSE AS BROWSE.action = 'browse', SELECT_IT AS SELECT_IT.action = 'select',
                   CHECKOUT AS CHECKOUT.action = 'checkout', REFUND AS REFUND.action = 'refund'
        )
        """,
    ),
    (
        "nested_quantifier_2m",
        "run_nested",
        """
        SELECT count(*) FROM run_nested MATCH_RECOGNIZE(
            ORDER BY i
            MEASURES count(*) AS len
            ONE ROW PER MATCH
            PATTERN ((A+)+ B)
            DEFINE A AS A.c = 'a', B AS B.c = 'b'
        )
        """,
    ),
]

checksums = {}
registered = set()
for name, tbl, sql in suite:
    if flt and flt not in name:
        continue
    if tbl not in registered:
        tdir = os.path.join(work_dir, tbl)
        os.makedirs(tdir, exist_ok=True)
        if not os.listdir(tdir):
            con.execute(f"COPY ({tables[tbl]}) TO '{tdir}' (FORMAT PARQUET, PER_THREAD_OUTPUT true)")
        con.execute(f"CREATE OR REPLACE VIEW {tbl} AS SELECT * FROM read_parquet('{tdir}/*.parquet')")
        registered.add(tbl)

    con.execute(sql).fetchall()  # 1 warmup
    runs_ms = []
    out = None
    for _ in range(3):
        t0 = time.perf_counter()
        out = con.execute(sql).fetchall()
        runs_ms.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(runs_ms)
    formatted = [[str(c) for c in row] for row in out]
    checksums[name] = formatted
    print(f"DUCKDB | `{name}` | wall={med:.1f} ms | out=({', '.join(formatted[0])})", flush=True)

with open(checksum_path, "w") as f:
    json.dump(checksums, f, indent=2)
PY

echo ""
echo "==> Step 3/3: Running DataFusion alone (threads=${THREADS})"
BENCH_DATA_DIR="${WORK_DIR}" \
BENCH_CHECKSUM_FILE="${WORK_DIR}/checksums.json" \
BENCH_THREADS="${THREADS}" \
BENCH_FILTER="${FILTER}" \
  "${REPO_DIR}/target/release/examples/bench_duckdb_suite"
