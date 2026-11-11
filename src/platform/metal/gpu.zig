//! Metal/MPS backend.
//!
//! Weight matrices are registered once at load time, which uploads them to a permanent
//! `MTLBuffer`; the GEMM entry point references that buffer instead of copying the weights on
//! every forward pass. Activations and outputs use per-thread scratch buffers in the C side.

const backend = @import("backend");

pub const has_gpu = true;
pub const name = "metal";

var state: u8 = 0;

extern fn mps_weight_register(data: [*]const f32, len: usize) ?*anyopaque;
extern fn mps_weight_release(handle: ?*anyopaque) void;

extern fn mps_sgemm_nt(
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

extern fn mps_sgemm_nn(
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
    return mps_weight_register(data.ptr, data.len);
}

fn releaseWeight(_: *anyopaque, handle: ?*anyopaque) void {
    mps_weight_release(handle);
}

fn sgemmNT(_: *anyopaque, args: backend.Gemm, weight: ?*anyopaque) bool {
    return mps_sgemm_nt(
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
    return mps_sgemm_nn(
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
    // MPS has a high per-call cost; see docs/plans/gpu-metal.md.
    .linear_min_ops = 1 << 28,
    .activation_min_ops = 1 << 28,
};

pub fn context() ?backend.Backend {
    return .{ .ptr = @ptrCast(&state), .vtable = &vtable };
}
