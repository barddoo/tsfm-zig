# tsfm-zig

Standalone inference for [Timestral](https://github.com/mistralai/timestral),
[Chronos-2](https://github.com/amazon-science/chronos-forecasting) and Chronos-Bolt checkpoints,
in Zig 0.17.1. One CLI and one Zig module; no Python at run time.

## Quickstart

Requires Zig 0.17.1.

```sh
zig build --release=fast

# model id: cached under ~/.cache/huggingface/hub, downloaded if missing
./zig-out/bin/timestral amazon/chronos-2 series.csv --horizon 96 > forecast.csv
```

`series.csv` is one line per variate, oldest value first. Values are comma- or
whitespace-separated; `nan`, `NA` and an empty field are missing. Separate series
with a blank line:

```csv
1,2,3,4,5
6,7,8,9,10

11,12,13,14,15
16,17,18,19,20
```

The output is CSV with a header, one row per series/variate/step/quantile:

```csv
series,variate,step,q0.01,q0.05,...,q0.99
0,0,1,-7.85,...
```

The model family is read from `config.json`. `<model>` may also be a local
directory containing `config.json` and `model.safetensors`.

## CLI reference

```
timestral <model> <series.csv> --horizon N [options]
```

`--horizon N` is required. Positional arguments are the model and the series
file.

| option              | meaning                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `--horizon N`       | steps to forecast (required)                                         |
| `--max-context N`   | keep only the most recent N steps (default: model context)           |
| `--block-patches N` | Timestral: patches decoded per block (default 32)                    |
| `--no-symmetric`    | Timestral: skip averaging with the reflected forecast                |
| `--no-clip`         | Timestral: do not clamp non-negative series at 0                     |
| `--past FILE`       | Timestral: past-only covariates                                      |
| `--known FILE`      | Timestral: known-future covariates                                   |
| `--cat-past FILE`   | Timestral: categorical past codes                                    |
| `--cat-known FILE`  | Timestral: categorical future codes                                  |
| `--prefetch`        | `madvise(MADV_WILLNEED)` the checkpoint after mapping                |
| `--revision REV`    | Hub branch, tag or commit (default `main`)                           |
| `--offline`         | never touch the network (also `HF_HUB_OFFLINE=1`)                    |
| `--threads N`       | worker threads (default: one per logical CPU; `1` = single-threaded) |
| `--time`            | print load and forecast wall time to stderr                          |

Environment: `HF_HUB_CACHE`, `HF_HOME`, `HF_TOKEN` (gated repos),
`HF_HUB_OFFLINE=1`.

Chronos-2 and Chronos-Bolt are univariate and reject `--past`, `--known`,
`--cat-past`, `--cat-known`, `--max-context`, `--block-patches`,
`--no-symmetric` and `--no-clip` with an explicit error. Chronos-2 horizons are
limited to 1024 steps.

## Input and output

`series.csv`: one block per series, one line per variate, equal row lengths.
Covariates use the same block layout:

| flag               | rows per block          | values per row          |
| ------------------ | ----------------------- | ----------------------- |
| `--past FILE`      | past-only covariates    | series length           |
| `--known FILE`     | known-future covariates | series length + horizon |
| `--cat-past FILE`  | categorical codes       | series length           |
| `--cat-known FILE` | categorical codes       | series length + horizon |

Categorical codes are target-encoded per series for a univariate target and
used raw otherwise, matching the Python package.

Output: `series,variate,step,q<level>...`, where `step` starts at 1 and the
quantile columns come from the model config.

## Models

|              | Timestral                                  | Chronos-2                     | Chronos-Bolt                                                            |
| ------------ | ------------------------------------------ | ----------------------------- | ----------------------------------------------------------------------- |
| architecture | decoder-only, patched                      | encoder-only, patched         | T5 encoder + one-token decoder                                          |
| targets      | univariate and multivariate                | univariate                    | univariate                                                              |
| covariates   | numeric/categorical, past and known-future | none in v1                    | none                                                                    |
| quantiles    | e.g. 9 (config)                            | e.g. 21 (config)              | e.g. 9 (config)                                                         |
| context      | config, `--max-context` override           | 8192, truncated from the left | 2048, truncated from the left                                           |
| horizon      | block-autoregressive, unbounded            | up to 1024 steps in one pass  | 64 per pass; longer horizons feed quantiles back, as the reference does |

Family detection: `architectures: ["Chronos2Model"]` → Chronos-2,
`["ChronosBoltModelForForecasting"]` → Chronos-Bolt; a config with Timestral
markers (`block_schedule`, `quantile_knots`, `patch_size`) → Timestral. A config
no registered family claims is an error, not a guess. Chronos-2 v1 covers the
univariate/singleton-group path; cross-series attention, known-future
covariates and autoregressive long-horizon unrolling are not ported.

A repo id is looked up in the Hugging Face cache and downloaded there if
missing. F32, F16 and BF16 Timestral checkpoints load; the published Chronos-2
checkpoint is F32. Weights are a read-only mmap used in place. `--prefetch`
turns cold page faults into a read-ahead at load; it changes only when pages are
read, never the values.

## Build options

| option       | values                                         | default                                    |
| ------------ | ---------------------------------------------- | ------------------------------------------ |
| `-Dblas=`    | `accelerate`, `openblas`, `builtin`            | `accelerate` on macOS, `builtin` elsewhere |
| `-Dgpu=`     | `metal` (macOS), `cuda` (Linux/NVIDIA), `none` | `none`                                     |
| `--release=` | `fast`, `safe`, `small`                        | debug                                      |
| `-Dtarget=`  | any Zig target                                 | host                                       |

`builtin` is about 1.5× slower than Accelerate on Timestral-28M (16 vs 10 ms on
an Apple Silicon laptop) but needs no C library and cross-compiles cleanly:

```sh
zig build --release=fast -Dblas=builtin -Dtarget=x86_64-linux-gnu
```

`-Dgpu=metal` requires a macOS target; `-Dgpu=cuda` requires a Linux target.

## GPU backends

GPU offload is opt-in because it keeps a second copy of the GEMM weights in
device memory (~111 MB for Timestral-28M, ~478 MB for Chronos-2).

```sh
zig build --release=fast -Dgpu=metal   # macOS, MPS
zig build --release=fast -Dgpu=cuda    # Linux, cuBLAS
```

- Only large products are offloaded. Dispatch thresholds are `2^28` ops on
  Metal for both linear and activation GEMMs, `2^20` (linear) / `2^24`
  (activation) on CUDA. Everything else stays on the CPU.
- Weights are uploaded once at load and referenced by `nn.Weight`; activations
  and results use per-thread scratch.
- Any failure (no device, missing library, allocation or command error) falls
  back to the CPU BLAS/SIMD path.
- CUDA is loaded with `dlopen`; no toolkit, headers or link-time library is
  needed. Set `TSFM_CUDA_LIB_DIR` to a private directory containing
  `libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13` (12.x also
  works). Only `libcuda.so` must be installed system-wide.

## Library API

`src/lib.zig` is the module entry point (named `timestral` in `build.zig`).
Consumers add it as a dependency and import it under their own name.

Top-level exports: `core`, `cfg`, `model`, `forecaster`, `safetensors`, `hub`,
`par`, `sys`, `chronos2`, `families`.

### `core.family`

```zig
const family = @import("core").family;

var req: family.Request = .{
    .io = io,
    .par_io = par_io,
    .allocator = allocator,
    .model_dir = model_dir,
    .config_text = config_text,
    .series_path = series_path,
    .horizon = 96,
};

const forecast = try families.Registry.run(&req); // family.Forecast
defer forecast.deinit(allocator);
```

| type             | members                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Request`        | `io`, `par_io`, `allocator`, `model_dir`, `config_text`, `series_path`, `covariates`, `horizon`, `options`, `prefetch` |
| `Options`        | `max_context: ?usize`, `block_patches: ?usize`, `no_symmetric: bool`, `no_clip: bool`                                  |
| `CovariatePaths` | `past`, `known`, `cat_past`, `cat_known` (optional paths)                                                              |
| `Forecast`       | `levels: []f64`, `values: []f32`, `series`, `variates`, `horizon`, `t_loaded`; `deinit(allocator)`                     |

`values` is `(series, variates, horizon, levels.len)` row-major. `Registry(.{…})`
checks each family at compile time and dispatches with an unrolled `inline for`.
`Registry.detect(allocator, config_text)` returns the family name or null.

### `core.series`

| item                                                   | purpose                                     |
| ------------------------------------------------------ | ------------------------------------------- |
| `Blocks{ data, blocks, rows, len }`                    | parsed `blocks × rows × len` float64 values |
| `parseBlocks(allocator, text)`                         | parse CSV text                              |
| `loadBlocks(io, allocator, path)`                      | read and parse a file                       |
| `toF32(allocator, values)`                             | convert `[]f64` to `[]f32`                  |
| `covariateFile(io, allocator, path, batch, len, what)` | load and shape-check a covariate file       |

### `core.safetensors`

```zig
var weights = try core.safetensors.Weights.open(io, allocator, path);
defer weights.close(io);
weights.prefetch(.will_need);              // optional
const w = try weights.get(io, "name", 4096); // []const f32, length-checked
```

F32 tensors are returned as slices into the mapping. F16/BF16 tensors are
converted to F32 on first access and cached. Invalid dtype, shape or offsets
return errors.

### `core.hub`

`resolve(io, allocator, env, spec, revision, offline) ![]u8` returns an owned
model directory: a local directory, a cached Hub repo, or a download into the
standard cache layout.

### Kernels and threading

- `core.nn`: `Weight` (host slice + optional device copy), `linear`, `gemmNT`,
  `gemmNN`, `rmsNorm`, `softmax`, `relu`, `siluInPlace`, `mulSilu`,
  `mulSigmoid`, `addInPlace`, `mulAddInPlace`, `ropeHalf`, `dot`,
  `clampNonfinite`, `lanes`, `defaultBackend`.
- `core.attn`: `Mask` (`full`, `keys_additive`, `causal`, `causal_keys`,
  `full_keys`), `Strides`, `selfAttend`, `selfAttendRuntime`, `maskedSoftmax`.
  `Mask` is a comptime parameter, so each variant is a mask-free loop.
- `core.par`: `threads`, `forRange`, `forRangeMin`, `both` over
  `std.Io.Threaded`.
- `core.resblock`: `ResBlock(act)` with `Act` = `.relu` or `.silu`, plus
  `Linear`.
- `core.sys`: `Mapped` (read-only file mmap) and `Advice.will_need`.

## Adding a model family

Families are registered at compile time in `src/families.zig`. A family is a
module with three declarations, checked by the compiler:

```zig
const core = @import("core");

pub const name = "My-Model";
pub fn matches(allocator: std.mem.Allocator, config_text: []const u8) !bool { ... }
pub fn run(req: *const core.family.Request) !core.family.Forecast { ... }
```

`run` loads weights, reads the series with `core.series`, forecasts, and returns
the quantile levels plus `(series, variates, horizon, levels)` values. Add the
module to the tuple in `src/families.zig`:

```zig
pub const Registry = core.family.Registry(.{ timestral, chronos2.runner, chronos_bolt.runner });
```

Run the attention tests with:

```sh
zig test --dep c -Mroot=src/core/attn.zig -Mc=src/core/c_none.zig
```

## Testing and parity

```sh
zig build test          # unit tests: attention, registry, series parser, Bolt pieces
```

`tests/reference.py`, `tests/reference_chronos2.py` and `tests/reference_bolt.py`
run the Python packages on the same inputs; `tests/compare.py` diffs results as
`|diff| / max(|reference|, 1)`.

Verified on the 28M Timestral model (~1e-6): single/multiple series,
multivariate targets, numeric and categorical covariates in every combination,
missing values, multi-block horizons, context caps, F16/BF16 checkpoints and the
builtin kernels; all model sizes 5M–209M matched on a plain forecast. Chronos-2
(`amazon/chronos-2`, snapshot `29ec3766`) matched 0–6e-6 on: irregular length,
NaN runs, constant series, independent series, >8192 context and a 1024-step
horizon. `--threads 1` and the default thread count are bit-identical.

Not verified: running on Linux or Windows (only cross-compiled) and the
OpenBLAS build.

## Performance

Warm Timestral-28M, `--horizon 96`, 512 context steps, load excluded. CUDA
numbers: RTX 3060 Laptop, i7-11800H; CPU is the builtin SIMD kernel (no
OpenBLAS installed).

| workload   | CPU t=1 | CUDA t=1 | CPU t=4 | CUDA t=4 |
| ---------- | ------- | -------- | ------- | -------- |
| 1 series   | 109 ms  | 66 ms    | 30 ms   | 59 ms    |
| 16 series  | 736 ms  | 160 ms   | 183 ms  | 110 ms   |
| 256 series | 12.8 s  | 2.50 s   | 3.49 s  | 1.14 s   |

| Chronos-2 workload     | CPU t=1 | CUDA t=1 | CPU t=4 | CUDA t=4 |
| ---------------------- | ------- | -------- | ------- | -------- |
| 1 series               | 388 ms  | 99 ms    | 105 ms  | 97 ms    |
| 16 series              | 2.88 s  | 237 ms   | 718 ms  | 215 ms   |
| 256 series             | 44.0 s  | 2.82 s   | 12.5 s  | 2.38 s   |
| 1 series, 8192 context | 2.64 s  | 325 ms   | —       | —        |

Metal is roughly CPU parity on an M4 after the weight cache: MPS's per-call
overhead is much higher than cuBLAS's. CUDA is the backend with a clear win.

`--prefetch` helps only cold files: with `--threads 1`, first-forecast time
dropped about 1.9× (Chronos-2 676→358 ms, Bolt-small 262→141 ms); with the
default pool it is a wash (Chronos-2 ~9% slower). Linux not measured.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The runner is a port
of the inference code in
[mistralai/timestral](https://github.com/mistralai/timestral) and
[amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting),
both Apache 2.0.

Model weights are licensed separately: Timestral weights are on the Hugging
Face Hub under a Modified MIT license (with a revenue cap); Chronos-2 weights
and code are Apache 2.0.
