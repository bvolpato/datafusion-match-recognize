# Contributing guide

Contributions are welcome. This guide covers licensing, local setup, and the checks expected on every pull request.

## Legal

### License

All original contributions to `datafusion-match-recognize` are licensed under the [Apache License, Version 2.0](./LICENSE).

### Developer Certificate of Origin (DCO)

Contributions are subject to the [Developer Certificate of Origin (DCO)](https://developercertificate.org/), included verbatim in [dco.txt](./dco.txt).

## Reporting an issue

Open an issue on GitHub with a minimal SQL reproducer, the observed output or error, and the expected SQL:2016 (or Trino) output.

## Local development

The crate currently points at a sibling checkout of Apache DataFusion (`../datafusion`) until `datafusion` 57 releases on crates.io.

```sh
# Format and lint
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings

# Verify optional proto feature compiles off
cargo check --lib --no-default-features

# Run all offline test suites (oracle fuzzer, SLT, invariance, disk spill, complexity)
cargo test

# Run examples and compile benchmarks
cargo run --example funnel
cargo run --example stock_vshape
cargo bench --no-run
```

To run the live differential oracle against Trino 484 locally:

```sh
docker run -d --name mr-trino-oracle -p 18080:8080 trinodb/trino:latest
cargo test --test trino_oracle -- --ignored --nocapture
docker rm -f mr-trino-oracle
```

## Pull request checklist

- Keep `MATCH_RECOGNIZE` on DataFusion public extension APIs without requiring a core fork.
- Add or update tests in the matching verification layer (`tests/oracle.rs`, `tests/slt/`, `tests/invariance.rs`, `tests/trino_oracle.rs`, or `tests/complexity.rs`).
- If a change adds or alters IR fields, update `proto/match_recognize.proto`, `src/proto.rs`, and the protobuf round-trip tests in `tests/invariance.rs`.
- If a change introduces or resolves a divergence against Trino 484, update [tests/divergences.md](./tests/divergences.md).
