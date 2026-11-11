//! Runtime interface shared by the GPU backends. The `gpu` module selects the implementation at
//! build time, but the rest of the code (and tests) only sees this vtable.
//!
//! A backend owns one opaque context (`ptr`) and a table of operations. The Metal and CUDA
//! implementations use a static context because they drive a single device; a mock in the tests
//! puts counters and thresholds in its own context. No global mutable backend state is required
//! by the callers.

/// Four-dimensional description of a GEMM. `a` is `m x k`, `b` is `n x k` for NT or `k x n` for
/// NN, and `out` is `m x n`; the leading dimensions are independent of the logical widths.
pub const Gemm = struct {
    m: usize,
    n: usize,
    k: usize,
    alpha: f32 = 1.0,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    out: []f32,
    ldc: usize,
};

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Upload a weight once. Returns an opaque device handle, or null on failure.
        register_weight: *const fn (ptr: *anyopaque, data: []const f32) ?*anyopaque,
        /// Release a handle from `register_weight`. Null is allowed.
        release_weight: *const fn (ptr: *anyopaque, handle: ?*anyopaque) void,
        /// `out = alpha * a * b^T`. Returns false when the caller must use the CPU kernel.
        sgemm_nt: *const fn (ptr: *anyopaque, args: Gemm, weight: ?*anyopaque) bool,
        /// `out = a * b`. Returns false when the caller must use the CPU kernel.
        sgemm_nn: *const fn (ptr: *anyopaque, args: Gemm) bool,
        /// Below this multiply-add count, use the CPU for `linear`.
        linear_min_ops: usize,
        /// Below this multiply-add count, use the CPU for activation products.
        activation_min_ops: usize,
    };

    pub fn registerWeight(self: Backend, data: []const f32) ?*anyopaque {
        return self.vtable.register_weight(self.ptr, data);
    }

    pub fn releaseWeight(self: Backend, handle: ?*anyopaque) void {
        self.vtable.release_weight(self.ptr, handle);
    }

    pub fn sgemmNT(self: Backend, args: Gemm, weight: ?*anyopaque) bool {
        return self.vtable.sgemm_nt(self.ptr, args, weight);
    }

    pub fn sgemmNN(self: Backend, args: Gemm) bool {
        return self.vtable.sgemm_nn(self.ptr, args);
    }

    pub inline fn linearMinOps(self: Backend) usize {
        return self.vtable.linear_min_ops;
    }

    pub inline fn activationMinOps(self: Backend) usize {
        return self.vtable.activation_min_ops;
    }
};
