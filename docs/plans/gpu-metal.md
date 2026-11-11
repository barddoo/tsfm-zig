# Metal/MPS backend and weight cache

**Status:** Implemented
**Date:** 2026-11-11
**Scope:** `-Dgpu=metal`, the `nn.Weight` weight cache, the C/MPS GEMM entry points, thresholds, lifetime, and the measurements behind them.

## 1. The problem

The first MPS wrapper built every input with `newBufferWithBytes` on every GEMM. For Timestral-28M
that re-uploaded roughly 100 MB of weights per forward pass: the GPU compute was fast, but a 29 ms
CPU forecast took 114 ms on the GPU, with about 85 ms spent copying weights.

The fix is not "cache anything whose pointer looks stable". A pointer into the safetensors mmap is
stable for the model's lifetime, while an activation pointer comes from an arena and is reused with
different contents. A heuristic in the GEMM wrapper would eventually cache the wrong buffer.

Instead, weights are explicitly registered at load time.

## 2. Design

### `nn.Weight` (src/core/nn.zig)

```zig
pub const Weight = struct {
    data: []const f32,          // mmap slice or model-owned permuted copy
    gpu: gpu.WeightBuffer,      // opaque MTLBuffer handle (null with gpu=none)

    pub fn init(data: []const f32) Weight;  // uploads once, for gpu=metal
    pub fn deinit(self: *const Weight) void; // releases the MTLBuffer
};
```

Every layer field that feeds a GEMM is a `Weight`: `Linear.w`, `Attn.qkv/out/gate`,
`Layer.w12/w3`, Chronos-2's `Attn.q/k/v/o`, Bolt's encoder and decoder projections. All are filled
in the model's `load` function, which is the explicit registration point. `nn.linear` takes the
`Weight` and passes `weight.gpu.handle` to `mps_sgemm_nt`; there is no registry lookup and no
pointer heuristic.

Timestral's fused QKV is the one non-mmap weight (it is permuted once at load). Its GPU buffer is
freed together with the host allocation in `Attn.deinit`.

Weights stay a read-only mmap in the CPU build. The docs' memory rule still holds: `-Dgpu=metal`
is opt-in because it deliberately duplicates the GEMM weights in private memory.

### C side (src/platform/metal/gemm.m)

- `mps_weight_register` creates one `MTLBuffer` per weight and returns a retained handle;
  `mps_weight_release` releases it when the model is deinited.
- Activations and C use per-thread scratch buffers (`pthread` TLS, released by a thread
  destructor), rounded up to 16 KiB and only grown. No `newBufferWithLength` per call.
- `MPSMatrixMultiplication` kernels are cached by `(m, n, k, alpha, transposeRight)`. Kernel
  construction can compile a pipeline state and only happens during warm-up.
- `beta = 0`, so the output buffer is fully overwritten; alpha is part of the kernel key.
- Every entry point returns `0` on success and non-zero on any failure. Zig then falls through to
  Accelerate or the builtin kernels, so a machine without Metal, a failed allocation, or a broken
  command buffer cannot produce wrong numbers.
- Submissions are serialized with a mutex. MPS's per-encode state is not documented as reentrant;
  the symmetric forward/mirrored passes run on two threads, and the GPU is serial anyway.

### Thresholds

`nn.linear` uses MPS only when `rows * out * in >= 2^28`. `gemmNT`/`gemmNN` (both operands are
activations, used by attention) use the same threshold. The CPU fallback for `linear` calls a
`gemmNTWeight` variant that never enters the GPU path: otherwise column-blocked CPU work would
re-upload the weight through the generic GEMM dispatch, which is exactly the original bug.

The threshold is high because MPS has a large per-call cost for small products. On an M4, an
empty command buffer costs ~15 us, but a 19x2048x384 GEMM costs ~0.4 ms per call through MPS
(0.21 ms when 64 are batched into one command buffer, so the cost is mostly encoding, not the
wait). Accelerate handles that product faster. Above the threshold the GPU wins or ties.

## 3. Measurements

Warm runs, M4, `--horizon 96`, 512 context steps. CPU is Accelerate. Load excluded.

| workload | CPU t=1 | Metal t=1 | CPU t=4 | Metal t=4 |
|---|---|---|---|---|
| Timestral-28M, 1 series | 36 ms | 25 ms | 14 ms | 12 ms |
| Timestral-28M, 16 series | 72 ms | 69 ms | 45 ms | 42 ms |
| Timestral-28M, 256 series | 1528 ms | 1685 ms | 869 ms | 732 ms |
| Chronos-2, 1 series | 50 ms | 35 ms | 31 ms | 31 ms |
| Chronos-2, 256 series | 1963 ms | 1985 ms | 1824 ms | 1758 ms |

Load time grows by the weight upload: about +28 ms for Timestral-28M (111 MB) and +70 ms for
Chronos-2 (478 MB).

The original upload-per-call implementation measured 459 ms for the 16-series Timestral workload
(CPU: 119 ms). With the same low threshold, the cache alone brings that to 85 ms; the tuned
threshold (CPU for small products) brings it to 42 ms.

## 4. Verification

- `zig build test` and `zig build test -Dgpu=metal` pass. The C source is attached to the `core`
  module so test executables link it once.
- Timestral-28M and Chronos-2 forecasts from `-Dgpu=metal` were diffed against the CPU build on
  single, 16 and 256-series inputs, including the 1-thread and 4-thread paths. The outputs were
  identical to the printed float32 precision in these runs; a deliberate 1e-3 perturbation of the
  GPU output confirmed the GPU path is on the critical path.
- MPS errors fall back to CPU rather than returning partial output.
