# Model-family registry

**Status:** Implemented
**Date:** 2026-11-11
**Scope:** How the runner chooses a model family from `config.json`, how a new family is added, and the decisions behind the shared pieces (attention kernel, weight loading). Families today: Timestral, Chronos-2, Chronos-Bolt.

## 1. Selection

`src/families.zig` lists the families as a comptime tuple and `core.family.Registry` dispatches over it with an unrolled `inline for`. Each family is a module with three declarations, checked by the compiler:

```zig
pub const name = "My-Model";
pub fn matches(allocator: std.mem.Allocator, config_text: []const u8) !bool
pub fn run(req: *const core.family.Request) !core.family.Forecast
```

A `config.json` that no family claims is `error.UnsupportedArchitecture`, with the supported names in the message. There is deliberately no default: the earlier "anything that is not Chronos-2 is Timestral" would load another family's checkpoint with the wrong assumptions.

| family | recognized by |
| --- | --- |
| Chronos-2 | `architectures` contains `Chronos2Model` |
| Chronos-Bolt | `architectures` contains `ChronosBoltModelForForecasting` |
| Timestral | no `architectures`; has `block_schedule`, `quantile_knots` and `patch_size` |

`core.family.hasArchitecture` does the `architectures` lookup and treats malformed configs (not an object, not an array) as "no match" instead of trapping.

## 2. Adding a family

1. Create `src/<family>/` with config parsing, the model, a forecaster and a `runner.zig` exposing the three declarations above.
2. Add the module to `build.zig` (it imports `core`) and the runner to the tuple in `src/families.zig`.
3. Verify against the family's reference implementation on real checkpoints (see `tests/reference_*.py` and `tests/compare.py`), including padded, gapped, all-missing and over-long inputs.

The runner owns loading and input handling; `main.zig` only parses arguments, dispatches and writes the CSV. Reject flags the family does not implement with an explicit error.

## 3. Shared pieces and why they look this way

- **Attention** (`core/attn.zig`): the mask is a comptime parameter (`full`, `keys_additive`, `causal`, `causal_keys`, `full_keys`), with an optional additive score bias for T5's relative positions. It is speed-neutral (the matrix products dominate); the point is one tested kernel for three models. Tests compare every mask, with and without bias, against a float64 reference.
- **Weights stay a read-only mmap, used in place.** Embedders such as `tsfm-pg` share the file's pages across processes, so private memory per process must stay at a few MB. Fusing q/k/v into one projection made Chronos-2 and Bolt 3-9% faster but copied 83 MB and 32 MB per process; writing q, k and v into one per-token buffer through a strided GEMM (no weight copy) matched the original speed exactly. Neither is kept: q, k and v stay separate. The GPU backends (`-Dgpu=metal`, `-Dgpu=cuda`, both off by default) are the deliberate exception: they upload GEMM weights once at load into device buffers, which costs one copy of the model's weights in private memory. They are opt-in for exactly that reason; see `docs/plans/gpu-metal.md` and `docs/plans/gpu-cuda.md`.
- **Prefetch** (`Weights.prefetch(.will_need)`, CLI `--prefetch`) is opt-in. On a cold file with one thread the first forecast is about 1.9x faster end to end (the read moves into load); with the default worker pool the unhinted parallel faults are already efficient and Chronos-2 got about 9% slower. `MADV_SEQUENTIAL` was slower in every case. Measured on macOS/APFS only.

## 4. Verification

`zig build test` runs the unit tests (attention masks, registry dispatch, series parsing, Bolt's relative-position buckets and quantile interpolation). Numerical parity against the PyTorch references needs checkpoints and is run by hand.
