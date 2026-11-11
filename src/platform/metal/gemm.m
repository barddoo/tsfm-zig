// Metal Performance Shaders sgemm for the Zig core.
//
// The cost model that shapes this file:
//
//   * A weight matrix is uploaded once, at load time, into a permanent MTLBuffer
//     (`mps_weight_register`, owned by the model and released in `mps_weight_release`).
//     `mps_sgemm_*` takes that buffer directly, so a forward pass never copies weights.
//   * Activations and outputs are per-thread, monotonically growing scratch buffers, so a call
//     only pays a memcpy plus the GPU work, not a fresh MTLBuffer allocation per GEMM.
//   * MPSMatrixMultiplication kernels are cached by shape (and alpha). Kernel construction can
//     compile a pipeline state, which is far more expensive than encoding the work.
//
// Every entry point returns 0 on success. On any failure (no Metal device, allocation failure,
// command-buffer error) it returns non-zero *without* producing an output buffer, so the Zig layer
// can fall back to BLAS/builtin kernels and stay correct.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static pthread_once_t g_device_once = PTHREAD_ONCE_INIT;

static void init_metal(void) {
    g_device = MTLCreateSystemDefaultDevice();
    if (g_device != nil) {
        g_queue = [g_device newCommandQueue];
    }
}

static int ensure_device(void) {
    pthread_once(&g_device_once, init_metal);
    return g_device != nil && g_queue != nil;
}

// ---------------------------------------------------------------------------------------------
// Per-thread scratch buffers

@interface MPSGemmScratch : NSObject
@property (nonatomic, strong) id<MTLBuffer> a;
@property (nonatomic, strong) id<MTLBuffer> b;
@property (nonatomic, strong) id<MTLBuffer> c;
@end

@implementation MPSGemmScratch
@end

static pthread_key_t g_scratch_key;
static pthread_once_t g_scratch_once = PTHREAD_ONCE_INIT;

static void destroy_scratch(void *ptr) {
    MPSGemmScratch *scratch = (__bridge_transfer MPSGemmScratch *)ptr;
    (void)scratch; // released when this scope ends
}

static void create_scratch_key(void) {
    pthread_key_create(&g_scratch_key, destroy_scratch);
}

static MPSGemmScratch *thread_scratch(void) {
    pthread_once(&g_scratch_once, create_scratch_key);
    MPSGemmScratch *scratch = (__bridge MPSGemmScratch *)pthread_getspecific(g_scratch_key);
    if (scratch == nil) {
        scratch = [MPSGemmScratch new];
        pthread_setspecific(g_scratch_key, (__bridge_retained void *)scratch);
    }
    return scratch;
}

/// Return `current` when it can hold `bytes`, else a new shared buffer rounded up to 16 KiB.
static id<MTLBuffer> ensure_buffer(id<MTLBuffer> current, size_t bytes) {
    if (current != nil && current.length >= bytes) {
        return current;
    }
    if (bytes < 4096) bytes = 4096;
    bytes = (bytes + 16383) & ~((size_t)16383);
    return [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

/// Upload `rows x cols` from a host matrix with row stride `pitch`, tightly packed on the device.
/// The copies are per row so padding columns are never read (attention matrices are slices of a
/// larger token buffer and can end at a guard page).
static id<MTLBuffer> upload_matrix(id<MTLBuffer> current, const float *src, size_t rows, size_t cols, size_t pitch) {
    const size_t bytes = rows * cols * sizeof(float);
    id<MTLBuffer> buf = ensure_buffer(current, bytes);
    if (buf == nil) return nil;
    float *dst = (float *)buf.contents;
    if (pitch == cols) {
        memcpy(dst, src, bytes);
    } else {
        for (size_t r = 0; r < rows; r++) {
            memcpy(dst + r * cols, src + r * pitch, cols * sizeof(float));
        }
    }
    return buf;
}

/// Copy `rows x cols` from a tightly packed device matrix into a host matrix with row stride
/// `pitch`. Only the used columns are written; padding stays untouched.
static void download_matrix(float *dst, const void *dev, size_t rows, size_t cols, size_t pitch) {
    const float *src = (const float *)dev;
    if (pitch == cols) {
        memcpy(dst, src, rows * cols * sizeof(float));
    } else {
        for (size_t r = 0; r < rows; r++) {
            memcpy(dst + r * pitch, src + r * cols, cols * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel cache
//
// MPSMatrixMultiplication is immutable once created, so one kernel can encode into any number of
// command buffers. The key is the complete shape plus alpha and the transpose flag; the cache is
// only touched during warm-up, then read-only.

typedef struct {
    int m, n, k;
    double alpha;
    int transpose_right;
    void *kernel; // retained MPSMatrixMultiplication
} KernelEntry;

#define MPS_KERNEL_CACHE_SIZE 256

static KernelEntry g_kernels[MPS_KERNEL_CACHE_SIZE];
static int g_kernel_count;
static pthread_mutex_t g_kernel_lock = PTHREAD_MUTEX_INITIALIZER;

/// MPS kernels and command buffers are fast to encode, but the framework's per-encode state is not
/// documented as reentrant. Serializing submission keeps two concurrent forward passes (the
/// symmetric-averaging pair) safe; the GPU is a single serial resource anyway.
static pthread_mutex_t g_submit_lock = PTHREAD_MUTEX_INITIALIZER;

static MPSMatrixMultiplication *kernel_for(int m, int n, int k, double alpha, int transpose_right) {
    pthread_mutex_lock(&g_kernel_lock);
    for (int i = 0; i < g_kernel_count; i++) {
        KernelEntry *e = &g_kernels[i];
        if (e->m == m && e->n == n && e->k == k && e->alpha == alpha &&
            e->transpose_right == transpose_right) {
            MPSMatrixMultiplication *kernel = (__bridge MPSMatrixMultiplication *)e->kernel;
            pthread_mutex_unlock(&g_kernel_lock);
            return kernel;
        }
    }

    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device
        transposeLeft:NO
        transposeRight:transpose_right ? YES : NO
        resultRows:(NSUInteger)m
        resultColumns:(NSUInteger)n
        interiorColumns:(NSUInteger)k
        alpha:alpha
        beta:0.0];
    if (kernel != nil && g_kernel_count < MPS_KERNEL_CACHE_SIZE) {
        KernelEntry *e = &g_kernels[g_kernel_count++];
        e->m = m;
        e->n = n;
        e->k = k;
        e->alpha = alpha;
        e->transpose_right = transpose_right;
        e->kernel = (__bridge_retained void *)kernel;
    }
    pthread_mutex_unlock(&g_kernel_lock);
    return kernel;
}

// ---------------------------------------------------------------------------------------------
// Weight buffers

void *mps_weight_register(const float *data, size_t len) {
    if (len == 0 || data == NULL || !ensure_device()) return NULL;
    @autoreleasepool {
        id<MTLBuffer> buf = [g_device newBufferWithBytes:data
                                                 length:len * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (buf == nil) return NULL;
        return (__bridge_retained void *)buf;
    }
}

void mps_weight_release(void *handle) {
    if (handle == NULL) return;
    // ARC releases the object at the end of this scope.
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle;
    (void)buf;
}

// ---------------------------------------------------------------------------------------------
// GEMM

static int mps_sgemm_impl(int m, int n, int k, double alpha,
                          const float *A, int lda,
                          const float *B, int ldb,
                          float *C, int ldc,
                          void *cached_b,
                          int transpose_right) {
    @autoreleasepool {
        if (m <= 0 || n <= 0 || k <= 0 || !ensure_device()) return -1;

        MPSGemmScratch *scratch = thread_scratch();
        if (scratch == nil) return -1;

        const size_t float_size = sizeof(float);
        const size_t b_rows = transpose_right ? (size_t)n : (size_t)k;
        const size_t b_cols = transpose_right ? (size_t)k : (size_t)n;
        // Device buffers are tightly packed; host matrices keep their own row strides.
        const size_t a_bytes = (size_t)m * (size_t)k * float_size;
        const size_t b_bytes = b_rows * b_cols * float_size;
        const size_t c_bytes = (size_t)m * (size_t)n * float_size;

        id<MTLBuffer> bufA = upload_matrix(scratch.a, A, (size_t)m, (size_t)k, (size_t)lda);
        if (bufA == nil) return -1;
        scratch.a = bufA;

        id<MTLBuffer> bufB;
        if (cached_b != NULL) {
            bufB = (__bridge id<MTLBuffer>)cached_b;
        } else {
            bufB = upload_matrix(scratch.b, B, b_rows, b_cols, (size_t)ldb);
            if (bufB == nil) return -1;
            scratch.b = bufB;
        }

        id<MTLBuffer> bufC = ensure_buffer(scratch.c, c_bytes);
        if (bufC == nil) return -1;
        scratch.c = bufC;

        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)m
            columns:(NSUInteger)k
            rowBytes:(NSUInteger)k * float_size
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)b_rows
            columns:(NSUInteger)b_cols
            rowBytes:(NSUInteger)b_cols * float_size
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)m
            columns:(NSUInteger)n
            rowBytes:(NSUInteger)n * float_size
            dataType:MPSDataTypeFloat32];

        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:dA];
        MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:dB];
        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:dC];
        if (mA == nil || mB == nil || mC == nil) return -1;

        MPSMatrixMultiplication *kernel = kernel_for(m, n, k, alpha, transpose_right);
        if (kernel == nil) return -1;

        pthread_mutex_lock(&g_submit_lock);
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        if (cmd == nil) {
            pthread_mutex_unlock(&g_submit_lock);
            return -1;
        }
        [kernel encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cmd commit];
        [cmd waitUntilCompleted];

        const int failed = cmd.status == MTLCommandBufferStatusError;
        if (failed) {
            fprintf(stderr, "MPS gemm %dx%dx%d failed: %s\n",
                    m, n, k, cmd.error.localizedDescription.UTF8String);
        } else {
            download_matrix(C, bufC.contents, (size_t)m, (size_t)n, (size_t)ldc);
        }
        pthread_mutex_unlock(&g_submit_lock);
        return failed ? -1 : 0;
    }
}

int mps_sgemm_nt(int m, int n, int k, float alpha,
                 const float *A, int lda,
                 const float *B, int ldb,
                 float *C, int ldc,
                 void *b_buf) {
    return mps_sgemm_impl(m, n, k, (double)alpha, A, lda, B, ldb, C, ldc, b_buf, 1);
}

int mps_sgemm_nn(int m, int n, int k,
                 const float *A, int lda,
                 const float *B, int ldb,
                 float *C, int ldc,
                 void *b_buf) {
    return mps_sgemm_impl(m, n, k, 1.0, A, lda, B, ldb, C, ldc, b_buf, 0);
}
