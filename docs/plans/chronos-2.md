# Add Chronos-2 to the Zig runner

**Status:** Implemented
**Date:** 2026-11-02
**Scope:** Run `amazon/chronos-2` on CPU in this repo, next to Timestral, sharing the weight loading, Hub download, BLAS/SIMD kernels and thread pool. Verified against the PyTorch reference to ~1e-6.

Reference source: local clone of `amazon-science/chronos-forecasting` at `1224faf`
(`src/chronos/chronos2/{model,layers,pipeline}.py`, plus `Patch` / `InstanceNorm` in `src/chronos/chronos_bolt.py`).
Checkpoint: `amazon/chronos-2`, snapshot `29ec3766`, cached locally under `~/.cache/huggingface/hub/models--amazon--chronos-2`.

## 1. Goal

Serija's bake-off found Chronos-2 best on lumpy demand and on the M5 cost proxy. Its server is Zig-only now, so Chronos-2 has to run in Zig to be used at all. This repo already has everything except the model itself.

Gate: only start once Serija's data check shows lumpy SKUs are a material share of inventory spend. The router that picks Chronos-2 per segment lives in Serija, not here.

## 2. The model, from the source

Checkpoint config (`config.json`):

| field | value |
| --- | --- |
| `d_model` / `d_kv` / `num_heads` | 768 / 64 / 12 |
| `d_ff`, activation | 3072, ReLU (not gated) |
| `num_layers` | 12 |
| `layer_norm_epsilon` | 1e-6 |
| `rope_theta` | 10000, full head dim (64) |
| `context_length` | 8192 |
| `input_patch_size` = `output_patch_size` = stride | 16 |
| `max_output_patches` | 64 (1024 steps per pass) |
| quantiles | 21: 0.01, 0.05, 0.10 … 0.95, 0.99 |
| `use_arcsinh`, `use_reg_token` | true, true |
| `time_encoding_scale` | 8192 |
| dtype | F32, 170 tensors, ~120M params (~480 MB) |

### Forward pass (inference only, dropout off)

1. **Mask:** `observed = !isnan(x)`.
2. **Truncate** context to the last 8192 steps.
3. **Instance norm** (in f32), over observed values only:
   `loc = nanmean(x)` (NaN → 0), `scale = sqrt(nanmean((x − loc)²))` (NaN → 1, 0 → 1e-5),
   then `x' = arcsinh((x − loc) / scale)`.
4. **Patch:** left-pad with NaN to a multiple of 16, split into non-overlapping patches of 16. Mask patches the same way (padding → 0). Zero the values wherever the mask is 0.
5. **Patch features**, 48 per patch, concatenated in this order:
   `[time_enc(16), values(16), mask(16)]`, with
   `time_enc = [−L, −L+1, …, −1] / 8192` where `L = num_patches × 16` (after padding).
6. **Context attention mask:** a patch is attended if any of its 16 steps is observed.
7. **Input embedding:** `ResidualBlock(48 → 3072 → 768)`:
   `out = W_o · relu(W_h · x + b_h) + b_o + (W_r · x + b_r)`. No layer norm.
8. **[REG] token:** append `shared.weight[1]` (row 1 of a `[2, 768]` table) after the context patches, attention mask 1.
9. **Future patches:** `n_out = ceil(horizon / 16)` patches, each with features
   `[time_enc(16), zeros(16), zeros(16)]`, `time_enc = [0, 1, …, n_out×16 − 1] / 8192`,
   embedded with the same input `ResidualBlock`; attention mask 1. (Known-future covariates would fill the middle 16 and set the mask; v1 sends none.)
10. **Sequence** = context patches, REG, future patches. `position_ids = 0 … seq_len − 1`.
11. **Encoder**, 12 blocks, each pre-norm with residuals:
    - **Time self-attention:** T5 RMS norm (`x · rsqrt(mean(x²) + eps) · w`, no mean, no bias) → q/k/v (no bias) → RoPE on q and k → attention → `o` → add.
    - **Group self-attention:** same, without RoPE, along the *batch* axis, masked so a series attends only to series with the same group id.
    - **Feed-forward:** norm → `wi` (768→3072) → ReLU → `wo` (3072→768) → add.
12. **Final norm** (`encoder.final_layer_norm`).
13. **Output head:** take the last `n_out` hidden states → `ResidualBlock(768 → 3072 → 336)`, then reshape each patch's 336 as `(21 quantiles × 16 steps)`, quantile-major.
14. **Un-scale:** `y = sinh(y') · scale + loc`. Truncate to `horizon`.

### Details that are easy to get wrong

- **Attention is unscaled.** Both backends use `scale = 1.0` (T5 convention). Do not divide by `sqrt(64)`.
- **Masks are additive `finfo(f32).min`**, not `−inf`. A fully masked row then softmaxes to uniform rather than NaN; match this.
- **RoPE uses `rotate_half`** (the first 32 dims are paired with the last 32), the Llama/HF layout, over the full 64-dim head. Check how `model.zig`'s Timestral RoPE pairs dimensions before reusing it; Timestral also rotates only part of the head (`rope_partial_factor`).
- **Time encoding is per step inside a patch**, not per patch, and starts at `−L` *after* left padding.
- **ResidualBlock linears have biases**; the attention and MLP linears do not.
- **Quantiles are not sorted by the model.** The PyTorch pipeline returns them as predicted. Keep that for parity; sort in the caller if needed.

### Group attention simplification

With no cross-learning each series is its own group (`group_ids = arange(batch)`). Group attention then attends over a single element, so softmax is 1 and the layer reduces to `x + o(v(norm(x)))` per token. Implement that fast path for v1 and verify it against the reference. Real cross-series attention (multivariate targets, covariates as group members) is a later step.

### Horizon

`horizon ≤ 1024` is one forward pass. Serija uses 90 days (6 output patches). Longer horizons need the pipeline's autoregressive unrolling (`_autoregressive_unroll_for_long_horizon`: expand the batch by 9 unrolled quantiles, append each to the context, and recombine with `weighted_quantile`). Out of scope for v1; return an error above 1024.

## 3. Repository layout

Do it in two commits so a regression can be told apart from a port bug.

### Commit 1: extract the shared core (no behavior change)

```
src/
  core/        safetensors.zig, hub.zig, nn.zig, par.zig, sys.zig, c_none.zig
  timestral/   model.zig, forecaster.zig, config.zig
  main.zig
  lib.zig
```

- `lib.zig` keeps re-exporting the same names (`cfg`, `model`, `forecaster`, `hub`, `par`, `sys`, `safetensors`), so Serija's server (`ts.model.Model.load`, `ts.forecaster.forecast`, `ts.hub.resolve`, …) builds unchanged.
- Rerun `tests/reference.py` + `tests/compare.py` for the cases in the README. They must still match to ~1e-6.

### Commit 2+: add Chronos-2

```
src/chronos2/
  config.zig       parse config.json + chronos_config
  model.zig        weights, encoder blocks, embeddings, head
  forecaster.zig   norm, patching, masks, sequence assembly, un-scaling
tests/
  reference_chronos2.py
```

- New exported module: `b.addModule("chronos2", ...)`, built on `core`. Export it from `lib.zig` as `chronos2`.
- Add to `core/nn.zig` what Chronos needs and Timestral lacks: ReLU, biased linear (if missing), T5 RMS norm without the f32 cast question (weights are F32 here), full-dim `rotate_half` RoPE.

## 4. Public API

Mirror the Timestral shape so the server can switch on model id:

```zig
pub const Model = struct {
    pub fn load(allocator: Allocator, weights: *const safetensors.Weights, config: Config) !Model;
    pub fn deinit(self: *Model, allocator: Allocator) void;
};

/// Returns `b × horizon × 21` quantiles, un-scaled, in the model's quantile order.
pub fn forecast(
    allocator: Allocator,
    io: std.Io,
    model: *const Model,
    inputs: []const f32,     // b × t, NaN = missing
    b: usize,
    t: usize,
    horizon: usize,
) ![]f32;
```

Batched series of different lengths are left-padded with NaN by the caller, as the PyTorch dataset does.

## 5. CLI

`timestral <model> series.csv --horizon N` picks the model from `config.json`: `architectures: ["Chronos2Model"]` → Chronos-2. (Originally anything else ran as Timestral; the family registry in [model-families.md](model-families.md) replaced that fallback with an error.) Same CSV in and out (`series,variate,step,q…`). Timestral-only flags (`--no-symmetric`, `--block-patches`, covariates) are rejected for Chronos-2 with a clear error in v1.

## 6. Verification

`tests/reference_chronos2.py`: same arguments and file formats as `tests/reference.py`, running `Chronos2Pipeline.from_pretrained("amazon/chronos-2").predict(...)` on CPU in f32 and writing all 21 quantiles. Install the reference from the local clone (`uv pip install -e ../chronos-forecasting`).

Diff with `tests/compare.py`. Target ~1e-6 scaled error, the same bar as Timestral. Cases:

1. one series, length 365, horizon 90 (Serija's shape)
2. length not a multiple of 16 (exercises left padding and the time encoding)
3. NaN runs in the middle and at the start (masks; Serija sends censored days as NaN)
4. a constant series (scale = 0 → 1e-5 path)
5. an all-zero intermittent series (Serija's no-sale SKUs)
6. several series in one file (independent groups; checks the group-attention fast path)
7. context longer than 8192 (truncation)
8. horizon exactly 1024; horizon 1025 errors
9. `--threads 1` vs default (output must be identical)

Also debug-compare intermediates on case 1 while porting: the 48-feature patch tensor, embeddings after the input block, hidden state after block 0, final hidden state.

## 7. Performance

- Each block: 4 × (768×768) attention matmuls + 2 × (768×3072) MLP matmuls per token. With 365 steps that is 23 context tokens + REG + 6 future = 30 tokens, so a forecast is cheap; the cost is the ~480 MB weight load, done once at server start via mmap.
- Parallelize like Timestral: linear layers by output columns, attention by (token, head).
- Record cold load and warm per-series time with `--time` on an M4 and on the Linux/OpenBLAS image, next to Timestral-28M (~9 ms warm with Accelerate).

## 8. Serija integration (separate change, in Serija)

- `--model chronos-2` in `services/forecast/zig/src/main.zig`; load both models at start if memory allows (~110 MB + ~480 MB).
- Convert the 21 quantiles to simulated paths exactly as for Timestral. Chronos-2 levels (0.01, 0.05, 0.10 … 0.95, 0.99) must be checked against what that conversion expects.
- Sort quantiles per step before path generation if they cross.
- Stockout correction stays in Serija's `censoring.zig`, before any model. Fully censored days go to Chronos-2 as NaN, which its masks handle natively.
- The model router (`auto` → per-segment choice by ADI/CV²) is Serija's job.

## 9. Licensing

- Chronos-2 weights and code: Apache-2.0. Add a NOTICE line crediting `amazon-science/chronos-forecasting` for the ported architecture.
- Timestral weights stay under their Modified MIT terms (revenue cap). The README should list each model's weight license separately.

## 10. Rename

Once Chronos-2 lands, rename the repo (proposed: `tsfm-zig`, package `.tsfm`). Keep exporting the `timestral` module name so dependents only change the URL and hash. Expect `zig build` to ask for a new `fingerprint` after the package name changes.

## 11. Out of scope for v1

- Long-horizon unrolling (> 1024 steps)
- Cross-series group attention (multivariate targets, covariates as group members)
- Known-future covariates through the future patches
- F16/BF16 checkpoints (the published one is F32)
- Embedding extraction (`Chronos2Pipeline.embed`)

## 12. Open questions

1. Does `model.zig`'s Timestral RoPE use the same dimension pairing as `rotate_half`, so the kernel can be shared, or does Chronos-2 need its own?
2. Is the group-attention fast path exact, or does the additive-mask path change results for fully masked tokens? Settle it with test case 3.
3. Should the server keep both models resident, or load Chronos-2 lazily on the first lumpy series?
