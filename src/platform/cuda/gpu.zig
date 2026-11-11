//! CUDA/cuBLAS backend.
//!
//! Weights are uploaded once at load time into permanent device allocations; activations are
//! copied through per-thread scratch buffers. The C side loads libcudart/libcublas with `dlopen`,
//! so building this backend requires no CUDA toolkit, headers or link-time libraries. If the
//! libraries or a device are unavailable every call returns false and the caller falls back to
//! the CPU kernels.

const backend = @import("backend");

pub const has_gpu = true;
pub const name = "cuda";

var state: u8 = 0;

extern fn cuda_weight_register(data: [*]const f32, len: usize) ?*anyopaque;
extern fn cuda_weight_release(handle: ?*anyopaque) void;

extern fn cuda_sgemm_nt(
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    C: [*]f32,
    ldc: c_int,
    b_buf: ?*anyopaque,
) c_int;

extern fn cuda_sgemm_nn(
    m: c_int,
    n: c_int,
    k: c_int,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    C: [*]f32,
    ldc: c_int,
    b_buf: ?*anyopaque,
) c_int;

fn registerWeight(_: *anyopaque, data: []const f32) ?*anyopaque {
    return cuda_weight_register(data.ptr, data.len);
}

fn releaseWeight(_: *anyopaque, handle: ?*anyopaque) void {
    cuda_weight_release(handle);
}

fn sgemmNT(_: *anyopaque, args: backend.Gemm, weight: ?*anyopaque) bool {
    return cuda_sgemm_nt(
        @intCast(args.m),
        @intCast(args.n),
        @intCast(args.k),
        args.alpha,
        args.a.ptr,
        @intCast(args.lda),
        args.b.ptr,
        @intCast(args.ldb),
        args.out.ptr,
        @intCast(args.ldc),
        weight,
    ) == 0;
}

fn sgemmNN(_: *anyopaque, args: backend.Gemm) bool {
    return cuda_sgemm_nn(
        @intCast(args.m),
        @intCast(args.n),
        @intCast(args.k),
        args.a.ptr,
        @intCast(args.lda),
        args.b.ptr,
        @intCast(args.ldb),
        args.out.ptr,
        @intCast(args.ldc),
        null,
    ) == 0;
}

const vtable: backend.Backend.VTable = .{
    .register_weight = registerWeight,
    .release_weight = releaseWeight,
    .sgemm_nt = sgemmNT,
    .sgemm_nn = sgemmNN,
    // cuBLAS is much cheaper per call than MPS.
    .linear_min_ops = 1 << 20,
    .activation_min_ops = 1 << 24,
};

pub fn context() ?backend.Backend {
    return .{ .ptr = @ptrCast(&state), .vtable = &vtable };
}
