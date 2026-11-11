# CUDA/cuBLAS backend

**Status:** Implemented
**Date:** 2026-11-11
**Scope:** `-Dgpu=cuda`, the runtime-loaded cuBLAS GEMM path, the shared `nn.Weight` cache, thresholds, verification on an RTX 3060.

## 1. Why cuBLAS + dlopen

The NVIDIA counterpart to the Metal/MPS backend is cuBLAS. It has the same shape as the existing
BLAS binding (row-major `sgemm` with transposes) and it parallelizes internally, so the layer code
does not need a CUDA-aware rewrite.

The host machine used for verification has the NVIDIA driver (`libcuda.so`) but no CUDA toolkit,
headers, or `nvcc`, and no root. The backend therefore loads `libcudart`/`libcublas` with `dlopen`
and declares only the handful of ABI-stable functions it needs. Consequences:

- `zig build -Dgpu=cuda` needs only a C compiler; no CUDA SDK.
- A standard CUDA installation is found in `/usr/local/cuda/lib64` or the loader path.
- A private copy can be selected with `TSFM_CUDA_LIB_DIR`; this is how the verification machine
  uses libs extracted from a container image.
- If the libraries or device are missing, initialization prints one warning and every GEMM returns
  non-zero so the CPU path runs. Correctness never depends on CUDA being present.

Only `libcuda.so` (the driver) must be installed system-wide.

## 2. Design

### Weight cache

`core.nn.Weight` is shared with the Metal backend. `WeightBuffer.init` calls
`cuda_weight_register`, which `cudaMalloc`s the weight and copies it once at model load. The handle
is the device pointer; `cuda_weight_release` frees it from `Model.deinit`.

### Per-thread context

cuBLAS handles are not shared across threads. Each host thread lazily creates:

- a `cublasHandle_t` (via `cublasCreate_v2`),
- a non-blocking `cudaStream_t`,
- device scratch buffers for A, B (activation-only products) and C, grown monotonically.

A `pthread_key_t` destructor releases all of them. Two concurrent forward passes (symmetric
averaging) therefore never share mutable state. Weight buffers are read-only and shared safely.

### GEMM and strides

For row-major `C[m,n] = alpha * A[m,k] * B[n,k]^T`, cuBLAS is called as

```
cublasSgemm(T, N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
```

which computes `C^T = B^T * A^T` in column-major. `gemmNN` uses `(N, N)`. This matches the mapping
already used by the CPU BLAS path.

The subtle part is strides: attention matrices are slices of larger token buffers, so `lda`, `ldb`
and `ldc` can exceed the number of columns. Copies use `cudaMemcpy2DAsync` with the row width
(`columns * 4`) and the full pitch (`ld * 4`). An initial implementation copied `rows * ld` floats;
that overwrote the padding columns of adjacent heads and only showed up for contexts long enough
to dispatch attention GEMMs to the GPU (8192 steps). The 2D copy is the fix.

### Thresholds

- `linear_min_ops = 1 << 20`: a registered weight, so only A/C move. cuBLAS is cheap enough per
  call to win from about a million multiply-adds.
- `activation_min_ops = 1 << 24`: attention products. Below this the per-call round trip is not
  worth it, especially when many heads call concurrently.

Both are in `src/platform/cuda/gpu.zig`.

## 3. Measurements

RTX 3060 Laptop (6 GB), i7-11800H (8C/16T), Ubuntu 26.04, driver 595.99.02. CPU is this project's
builtin SIMD kernels (no OpenBLAS installed), warm runs, `--horizon 96`, 512 context steps.

| workload | CPU t=1 | CUDA t=1 | CPU t=4 | CUDA t=4 |
|---|---|---|---|---|
| Timestral-28M, 1 series | 109 ms | 66 ms | 30 ms | 59 ms |
| Timestral-28M, 16 series | 736 ms | 160 ms | 183 ms | 110 ms |
| Timestral-28M, 256 series | 12.8 s | 2.50 s | 3.49 s | 1.14 s |
| Chronos-2, 1 series | 388 ms | 99 ms | 105 ms | 97 ms |
| Chronos-2, 16 series | 2.88 s | 237 ms | 718 ms | 215 ms |
| Chronos-2, 256 series | 44.0 s | 2.82 s | 12.5 s | 2.38 s |
| Chronos-2, 1 series, 8192 context | 2.64 s | 325 ms | — | — |

CUDA is effectively flat across `--threads` because cuBLAS uses the device internally. The
one-series Timestral case is the exception where CPU wins at 4 threads: the matrices are small
enough that the device round trip dominates. Load time grows by the one-time weight upload
(~0.17 s for Timestral-28M, ~0.23 s for Chronos-2, on this PCIe link).

## 4. Verification

- `zig build test -Dgpu=cuda` passes on the RTX 3060 host.
- Forecasts were compared against the CPU build with `tests/compare.py` on 1, 16 and 256 series,
  for Timestral-28M and Chronos-2, at 1/4/8 threads, plus a Chronos-2 8192-step context. Max
  absolute error was 8e-6 and max scaled error 7.6e-7.
- GPU output is identical for `--threads 1` and `--threads 8`, so dispatch does not depend on the
  thread count.
- A standalone C test compared `cuda_sgemm_nn`/`nt` against a naive CPU implementation with
  padded leading dimensions; both layouts agree to float rounding. The 2D-copy fix was validated
  against the 8192-context case that originally exposed it.

## 5. Running on the verification host

The libs are extracted from the Ollama container layer:

```bash
mkdir -p ~/.local/lib/tsfm-cuda
D=~/.local/share/docker/containerd/.../usr/lib/ollama/cuda_v13
cp -a $D/libcudart.so* $D/libcublas.so* $D/libcublasLt.so* ~/.local/lib/tsfm-cuda/
zig build --release=fast -Dgpu=cuda
TSFM_CUDA_LIB_DIR=$HOME/.local/lib/tsfm-cuda ./zig-out/bin/timestral ...
```
