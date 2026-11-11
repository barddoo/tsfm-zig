# tsfm-zig (Zig runner)

Standalone CPU inference for [Timestral](https://github.com/mistralai/timestral),
[Chronos-2](https://github.com/amazon-science/chronos-forecasting) and Chronos-Bolt checkpoints, in Zig 0.17.
Everything except the matrix products is Zig standard library: files and memory-mapped weights
(`std.Io`), the Hub download (`std.http.Client`, `std.json`), threads (`std.Io.Threaded`) and SIMD
(`@Vector`). Matrix products can run on a C BLAS (Apple Accelerate by default on macOS, OpenBLAS
optionally) bound through [translate-c](https://codeberg.org/ziglang/translate-c) (pinned to its
`zig-0.17.x` branch); the `builtin` SIMD kernels need no C library at all. Elementwise work (softmax,
SiLU, ReLU, RMSNorm, RoPE, residual adds) uses `@Vector`, and the layers run on a
`std.Io.Threaded` pool: linear layers split by output columns, attention by (row, head), and the
forward and mirrored passes of symmetric averaging run concurrently. Attention keeps Q/K/V in one
`(token, q|k|v, head, head_dim)` buffer: the fused projection rows are reordered once when the
weights load, so every head is a contiguous vector and the time and variate kernels read it without
per-token gathers (the variate path also hoists its key mask out of the per-head loop). Output is
identical for any thread count.

```bash
zig build --release=fast
./zig-out/bin/timestral mistralai/Timestral-28M series.csv --horizon 96 > forecast.csv
./zig-out/bin/timestral amazon/chronos-2 series.csv --horizon 96 > forecast.csv
./zig-out/bin/timestral amazon/chronos-bolt-small series.csv --horizon 96 > forecast.csv
```

`<model>` is a directory holding `config.json` and `model.safetensors`, or a Hub repo id. The model
family is read from `config.json`: `architectures: ["Chronos2Model"]` runs Chronos-2, `["ChronosBoltModelForForecasting"]` runs Chronos-Bolt and a Timestral
config (recognized by `block_schedule`, `quantile_knots` and `patch_size`) runs Timestral; a config no
registered family claims is an error. A repo id is looked up in the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME`,
`~/.cache/huggingface/hub`) and downloaded there if missing (`HF_TOKEN` is used for gated repos;
`--offline` or `HF_HUB_OFFLINE=1` forbids the network). F32, F16 and BF16 Timestral checkpoints
load; the published Chronos-2 checkpoint is F32.

Weights are a read-only mmap used in place, so a file that is not in the page cache is read by page
faults during the first forecast. `--prefetch` (library: `Weights.prefetch(.will_need)`) asks the
kernel to read it in at load instead (`madvise(MADV_WILLNEED)`). Measured on macOS with a cold file,
the first forecast with `--threads 1` is about 1.9x faster end to end (Chronos-2 676 to 358 ms,
Bolt-small 262 to 141 ms); with the default worker pool the unhinted parallel faults are already
efficient, so the gain is small and Chronos-2 got about 9% slower. It never changes the output and
costs nothing on a cached file. Linux has not been measured.

## Adding a model family

Families are registered at compile time in `src/families.zig`. A family is a module with three
declarations, checked against `core.family` by the compiler:

```zig
pub const name = "My-Model";
pub fn matches(allocator: std.mem.Allocator, config_text: []const u8) !bool { ... }
pub fn run(req: *const core.family.Request) !core.family.Forecast { ... }
```

`run` loads the weights, reads the series with `core.series`, forecasts, and returns the quantile
levels and a `(series, variates, horizon, levels)` array; the CLI does argument parsing and CSV
output. Then add the module to the tuple in `src/families.zig`. Shared attention goes through
`core.attn`, whose `Mask` (`full`, `keys_additive`, `causal`, `causal_keys`, `full_keys`) is a comptime
parameter, so each variant compiles to its own mask-free loop. Run its tests with
`zig test --dep c -Mroot=src/core/attn.zig -Mc=src/core/c_none.zig`.

## Model support

| | Timestral | Chronos-2 | Chronos-Bolt |
|---|---|---|---|
| architecture | decoder-only, patched | encoder-only, patched | T5 encoder + one-token decoder, patched |
| targets | univariate and multivariate | univariate (each row is independent) | univariate (each row is independent) |
| covariates | numeric and categorical, past and known-future | none in v1 | none |
| quantiles | 9 (config-defined) | 21 (config-defined) | 9 (config-defined) |
| context | config-defined, `--max-context` override | 8192, truncated from the left | 2048, truncated from the left |
| horizon | block-autoregressive, unbounded | up to 1024 steps in one pass | 64 steps per pass; longer horizons feed each quantile back and re-reduce, as the reference does |

Chronos-2 v1 is the univariate/singleton-group path: cross-series (group) attention, known-future
covariates and the autoregressive long-horizon unrolling are out of scope. `--past`, `--known`,
`--cat-past`, `--cat-known`, `--max-context`, `--block-patches`, `--no-symmetric` and `--no-clip`
are rejected for Chronos-2 with an explicit error.

## Input

`series.csv` has one line per variate; values are comma or whitespace separated and `nan` / `NA` / an
empty field marks a missing value. **Several series** go in one file, separated by a blank line (all
with the same number of variates and steps). Covariates use the same blocks, one block per series and
one line per covariate:

| flag | rows | steps per line |
|---|---|---|
| `--past FILE` | past-only covariates | as the series |
| `--known FILE` | known-future covariates | series + horizon |
| `--cat-past FILE` | categorical codes (numbers) | as the series |
| `--cat-known FILE` | categorical codes (numbers) | series + horizon |

Categorical codes are target-encoded per series for a univariate target and used raw otherwise, as in
the Python package. The covariate flags are Timestral-only. Output is CSV:
`series,variate,step,q<level>...`.

Other options: `--threads N` (default: one per logical CPU; `1` is single-threaded), `--max-context N`, `--block-patches N`, `--no-symmetric`, `--no-clip`, `--revision REV`,
`--time` (load and forecast wall time on stderr), `--offline`.

## Build options

| option | values | default |
|---|---|---|
| `-Dblas=` | `accelerate`, `openblas`, `builtin` | `accelerate` on macOS, `builtin` elsewhere |

`builtin` is about 1.5x slower than Accelerate on the 28M Timestral model (16 vs 10 ms on an
Apple Silicon laptop); use `-Dblas=openblas` on Linux for speed. The builtin kernels tile the
inner product 4 rows x 4 columns (2 rows on wide AVX2 targets) so each streamed row of the weight
matrix is reused across several output rows. Cross-compiling works with the builtin kernels (Linux
x86_64/aarch64, musl, Windows, macOS):
`zig build --release=fast -Dblas=builtin -Dtarget=x86_64-linux-gnu`.

## Checked against PyTorch

`tests/reference.py` and `tests/reference_chronos2.py` take the same arguments and file formats and
run the Python packages; `tests/compare.py` diffs two result files. The error is
`|diff| / max(|reference|, 1)`.

Timestral: on the 28M model, with `uv sync --frozen` for PyTorch, the Zig output matched to about
1e-6 for: a single series, several series, multivariate targets, past, known, numeric and categorical
covariates in every combination, missing values, multi-block horizons, a context cap, fp16 and bf16
checkpoints, and the builtin kernels. All four model sizes (5M to 209M) matched on a plain forecast.

Chronos-2 (`amazon/chronos-2`, snapshot `29ec3766`): matched the `Chronos2Pipeline` reference with
scaled error from 0 to 6e-6 (about 1e-6 on most cases) on: a 365-step series with a 90-step horizon;
a length that is not a multiple of 16; NaN runs at the start and in the middle; a constant series
(the scale-0 path); an intermittent all-zero-leaning series; several independent series in one file;
a context longer than 8192 (left truncation); and a 1024-step horizon (1025 is rejected).
`--threads 1` and the default thread count give bit-identical output.

Not verified: running on Linux or Windows (only cross-compiled), and the OpenBLAS build.

## License

The runner is Apache License 2.0, see [LICENSE](LICENSE), and [NOTICE](NOTICE) for attribution. It is
a port of the inference code in [mistralai/timestral](https://github.com/mistralai/timestral) and
[amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting), both
Apache 2.0.

Model weights are licensed separately: Timestral weights are distributed on the Hugging Face Hub
under a Modified MIT license (with a revenue cap), available in each model repository; Chronos-2
weights and code are Apache 2.0.
