// CUDA/cuBLAS sgemm for the Zig core, loaded at runtime with dlopen.
//
// Why dlopen: the build then needs no CUDA toolkit, headers or link-time libraries. If
// libcudart/libcublas are not installed (or there is no device) every entry point returns
// non-zero and the Zig layer falls back to the CPU kernels. A normal CUDA installation is found
// in the standard locations; a private directory can be pointed at with TSFM_CUDA_LIB_DIR.
//
// Cost model matches the Metal backend:
//   * weights are uploaded once at load time and live in device memory for the model's lifetime;
//   * activations/results go through per-thread pinned-free scratch buffers;
//   * one cuBLAS handle and one non-blocking stream per host thread.
//
// cuBLAS is column-major. For a row-major C[m,n] = A[m,k] * B[n,k]^T, call it with the arguments
// swapped and B transposed:
//     cublasSgemm(T, N, n, m, k, ..., B, ldb, A, lda, ..., C, ldc)
// which computes (C)^T = (B)^T * (A)^T in column-major. The NN variant uses (N, N) instead.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
typedef int cublasStatus_t;
typedef void *cudaStream_t;
typedef void *cublasHandle_t;

typedef cudaError_t (*cudaMalloc_fn)(void **, size_t);
typedef cudaError_t (*cudaFree_fn)(void *);
typedef cudaError_t (*cudaMemcpy_fn)(void *, const void *, size_t, int);
typedef cudaError_t (*cudaMemcpyAsync_fn)(void *, const void *, size_t, int, cudaStream_t);
typedef cudaError_t (*cudaMemcpy2DAsync_fn)(void *, size_t, const void *, size_t, size_t, size_t, int, cudaStream_t);
typedef cudaError_t (*cudaStreamCreateWithFlags_fn)(cudaStream_t *, unsigned int);
typedef cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t);
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t);
typedef cudaError_t (*cudaSetDevice_fn)(int);
typedef cudaError_t (*cudaGetDeviceCount_fn)(int *);

typedef cublasStatus_t (*cublasCreate_fn)(cublasHandle_t *);
typedef cublasStatus_t (*cublasDestroy_fn)(cublasHandle_t);
typedef cublasStatus_t (*cublasSetStream_fn)(cublasHandle_t, cudaStream_t);
typedef cublasStatus_t (*cublasSgemm_fn)(cublasHandle_t, int, int, int, int, int,
                                         const float *, const float *, int,
                                         const float *, int,
                                         const float *, float *, int);

static struct {
    cudaMalloc_fn alloc;
    cudaFree_fn release;
    cudaMemcpy_fn copy;
    cudaMemcpyAsync_fn copy_async;
    cudaMemcpy2DAsync_fn copy_2d_async;
    cudaStreamCreateWithFlags_fn stream_create;
    cudaStreamDestroy_fn stream_destroy;
    cudaStreamSynchronize_fn stream_sync;
    cudaSetDevice_fn set_device;
    cudaGetDeviceCount_fn device_count;

    cublasCreate_fn create;
    cublasDestroy_fn destroy;
    cublasSetStream_fn set_stream;
    cublasSgemm_fn sgemm;
} g_api;

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static int g_ready;
static int g_device;

enum { CUDA_MEMCPY_HOST_TO_DEVICE = 1, CUDA_MEMCPY_DEVICE_TO_HOST = 2 };
enum { CUDA_STREAM_NON_BLOCKING = 1 };
enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };

static void *dlopen_path(const char *dir, const char *name) {
    char path[4096];
    if (dir != NULL && dir[0] != '\0') {
        int n = snprintf(path, sizeof(path), "%s/%s", dir, name);
        if (n > 0 && (size_t)n < sizeof(path)) {
            void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (h != NULL) return h;
        }
    }
    return dlopen(name, RTLD_NOW | RTLD_GLOBAL);
}

static void *load_lib(const char *const *names, size_t count) {
    const char *env = getenv("TSFM_CUDA_LIB_DIR");
    const char *home = getenv("HOME");
    char home_dir[4096] = {0};
    if (home != NULL) snprintf(home_dir, sizeof(home_dir), "%s/.local/lib/tsfm-cuda", home);
    const char *dirs[] = { env, home_dir, "/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", NULL };

    for (size_t n = 0; n < count; n++) {
        for (size_t d = 0; dirs[d] != NULL; d++) {
            void *h = dlopen_path(dirs[d], names[n]);
            if (h != NULL) return h;
        }
    }
    return NULL;
}

#define LOAD_SYM(var, name)                              \
    do {                                                 \
        *(void **)(&g_api.var) = dlsym(rt, name);        \
        if (g_api.var == NULL) {                         \
            *(void **)(&g_api.var) = dlsym(blas, name);  \
        }                                                \
    } while (0)

static void init_cuda(void) {
    static const char *const cudart_names[] = { "libcudart.so.13", "libcudart.so.12", "libcudart.so" };
    static const char *const cublaslt_names[] = { "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so" };
    static const char *const cublas_names[] = { "libcublas.so.13", "libcublas.so.12", "libcublas.so" };

    void *rt = load_lib(cudart_names, sizeof(cudart_names) / sizeof(cudart_names[0]));
    // Load the Lt library first so cuBLAS's DT_NEEDED resolves from the same private directory.
    load_lib(cublaslt_names, sizeof(cublaslt_names) / sizeof(cublaslt_names[0]));
    void *blas = load_lib(cublas_names, sizeof(cublas_names) / sizeof(cublas_names[0]));
    if (rt == NULL || blas == NULL) {
        fprintf(stderr, "cuda backend unavailable: set TSFM_CUDA_LIB_DIR to the CUDA lib directory\n");
        return;
    }

    LOAD_SYM(alloc, "cudaMalloc");
    LOAD_SYM(release, "cudaFree");
    LOAD_SYM(copy, "cudaMemcpy");
    LOAD_SYM(copy_async, "cudaMemcpyAsync");
    LOAD_SYM(copy_2d_async, "cudaMemcpy2DAsync");
    LOAD_SYM(stream_create, "cudaStreamCreateWithFlags");
    LOAD_SYM(stream_destroy, "cudaStreamDestroy");
    LOAD_SYM(stream_sync, "cudaStreamSynchronize");
    LOAD_SYM(set_device, "cudaSetDevice");
    LOAD_SYM(device_count, "cudaGetDeviceCount");
    LOAD_SYM(create, "cublasCreate_v2");
    LOAD_SYM(destroy, "cublasDestroy_v2");
    LOAD_SYM(set_stream, "cublasSetStream_v2");
    LOAD_SYM(sgemm, "cublasSgemm_v2");

    if (g_api.alloc == NULL || g_api.release == NULL || g_api.copy == NULL ||
        g_api.copy_async == NULL || g_api.copy_2d_async == NULL ||
        g_api.stream_create == NULL || g_api.stream_destroy == NULL ||
        g_api.stream_sync == NULL || g_api.set_device == NULL || g_api.create == NULL ||
        g_api.destroy == NULL || g_api.set_stream == NULL || g_api.sgemm == NULL) {
        fprintf(stderr, "cuda backend unavailable: missing CUDA symbols\n");
        return;
    }

    int count = 0;
    if (g_api.device_count(&count) != 0 || count <= 0) {
        fprintf(stderr, "cuda backend unavailable: no CUDA device\n");
        return;
    }
    if (g_api.set_device(0) != 0) {
        fprintf(stderr, "cuda backend unavailable: cannot select device 0\n");
        return;
    }
    g_device = 0;
    g_ready = 1;
}

static int ensure_cuda(void) {
    pthread_once(&g_init_once, init_cuda);
    return g_ready;
}

// ---------------------------------------------------------------------------------------------
// Per-thread cuBLAS handle, stream and device scratch

typedef struct {
    cublasHandle_t handle;
    cudaStream_t stream;
    void *a;
    size_t a_cap;
    void *b;
    size_t b_cap;
    void *c;
    size_t c_cap;
} CudaCtx;

static pthread_key_t g_ctx_key;
static pthread_once_t g_ctx_once = PTHREAD_ONCE_INIT;

static void destroy_ctx(void *ptr) {
    CudaCtx *ctx = (CudaCtx *)ptr;
    if (ctx == NULL) return;
    if (ctx->handle != NULL && g_api.destroy != NULL) g_api.destroy(ctx->handle);
    if (ctx->stream != NULL && g_api.stream_destroy != NULL) g_api.stream_destroy(ctx->stream);
    if (ctx->a != NULL) g_api.release(ctx->a);
    if (ctx->b != NULL) g_api.release(ctx->b);
    if (ctx->c != NULL) g_api.release(ctx->c);
    free(ctx);
}

static void create_ctx_key(void) {
    pthread_key_create(&g_ctx_key, destroy_ctx);
}

static CudaCtx *thread_ctx(void) {
    pthread_once(&g_ctx_once, create_ctx_key);
    CudaCtx *ctx = (CudaCtx *)pthread_getspecific(g_ctx_key);
    if (ctx != NULL) return ctx;

    ctx = (CudaCtx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) return NULL;
    if (g_api.set_device(g_device) != 0) goto fail;
    if (g_api.create(&ctx->handle) != 0) goto fail;
    if (g_api.stream_create(&ctx->stream, CUDA_STREAM_NON_BLOCKING) != 0) goto fail;
    if (g_api.set_stream(ctx->handle, ctx->stream) != 0) goto fail;
    pthread_setspecific(g_ctx_key, ctx);
    return ctx;

fail:
    destroy_ctx(ctx);
    return NULL;
}

static int ensure_buf(void **slot, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 0;
    void *next = NULL;
    if (g_api.alloc(&next, bytes) != 0 || next == NULL) return -1;
    if (*slot != NULL) g_api.release(*slot);
    *slot = next;
    *cap = bytes;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Weight buffers

void *cuda_weight_register(const float *data, size_t len) {
    if (len == 0 || data == NULL || !ensure_cuda()) return NULL;
    void *device = NULL;
    size_t bytes = len * sizeof(float);
    if (g_api.alloc(&device, bytes) != 0 || device == NULL) return NULL;
    if (g_api.copy(device, data, bytes, CUDA_MEMCPY_HOST_TO_DEVICE) != 0) {
        g_api.release(device);
        return NULL;
    }
    return device;
}

void cuda_weight_release(void *handle) {
    if (handle == NULL || g_api.release == NULL) return;
    g_api.release(handle);
}

// ---------------------------------------------------------------------------------------------
// GEMM

static int cuda_sgemm_impl(int m, int n, int k, float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float *C, int ldc,
                           void *cached_b,
                           int transpose_right) {
    if (m <= 0 || n <= 0 || k <= 0 || !ensure_cuda()) return -1;
    CudaCtx *ctx = thread_ctx();
    if (ctx == NULL) return -1;

    const size_t a_bytes = (size_t)m * (size_t)lda * sizeof(float);
    const size_t b_bytes = transpose_right
        ? (size_t)n * (size_t)ldb * sizeof(float)
        : (size_t)k * (size_t)ldb * sizeof(float);
    const size_t c_bytes = (size_t)m * (size_t)ldc * sizeof(float);

    if (ensure_buf(&ctx->a, &ctx->a_cap, a_bytes) != 0) return -1;
    if (ensure_buf(&ctx->c, &ctx->c_cap, c_bytes) != 0) return -1;

    const void *dev_b;
    if (cached_b != NULL) {
        dev_b = cached_b;
    } else {
        if (ensure_buf(&ctx->b, &ctx->b_cap, b_bytes) != 0) return -1;
        const size_t b_rows = transpose_right ? (size_t)n : (size_t)k;
        const size_t b_cols = transpose_right ? (size_t)k : (size_t)n;
        if (g_api.copy_2d_async(ctx->b, (size_t)ldb * sizeof(float), B, (size_t)ldb * sizeof(float),
                                b_cols * sizeof(float), b_rows,
                                CUDA_MEMCPY_HOST_TO_DEVICE, ctx->stream) != 0) {
            return -1;
        }
        dev_b = ctx->b;
    }

    if (g_api.copy_2d_async(ctx->a, (size_t)lda * sizeof(float), A, (size_t)lda * sizeof(float),
                            (size_t)k * sizeof(float), (size_t)m,
                            CUDA_MEMCPY_HOST_TO_DEVICE, ctx->stream) != 0) {
        return -1;
    }

    float a = alpha;
    float beta = 0.0f;
    cublasStatus_t status;
    if (transpose_right) {
        // (C)^T = (B)^T * (A)^T, both operands row-major.
        status = g_api.sgemm(ctx->handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                             &a, (const float *)dev_b, ldb, (const float *)ctx->a, lda,
                             &beta, (float *)ctx->c, ldc);
    } else {
        status = g_api.sgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &a, (const float *)dev_b, ldb, (const float *)ctx->a, lda,
                             &beta, (float *)ctx->c, ldc);
    }
    if (status != 0) return -1;
    if (g_api.stream_sync(ctx->stream) != 0) return -1;
    if (g_api.copy_2d_async(C, (size_t)ldc * sizeof(float), ctx->c, (size_t)ldc * sizeof(float),
                            (size_t)n * sizeof(float), (size_t)m,
                            CUDA_MEMCPY_DEVICE_TO_HOST, ctx->stream) != 0) {
        return -1;
    }
    if (g_api.stream_sync(ctx->stream) != 0) return -1;
    return 0;
}

int cuda_sgemm_nt(int m, int n, int k, float alpha,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc,
                  void *b_buf) {
    return cuda_sgemm_impl(m, n, k, alpha, A, lda, B, ldb, C, ldc, b_buf, 1);
}

int cuda_sgemm_nn(int m, int n, int k,
                  const float *A, int lda,
                  const float *B, int ldb,
                  float *C, int ldc,
                  void *b_buf) {
    return cuda_sgemm_impl(m, n, k, 1.0f, A, lda, B, ldb, C, ldc, b_buf, 0);
}
