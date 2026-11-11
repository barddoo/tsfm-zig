//! Tests for the GPU backend interface.
//!
//! There are two layers:
//!   1. a hardware test that runs when `-Dgpu=metal`/`cuda` found a device, and is skipped
//!      otherwise;
//!   2. a mock backend that exercises the dispatch, cache and fallback logic without a GPU.
//!
//! The matrices use padded leading dimensions, matching attention slices. Besides the numerical
//! result, the hardware test asserts that the unused columns of the output keep their sentinel
//! values: a flat `rows * ldc` copy (the original bug) overwrites the padding of adjacent heads.

const std = @import("std");
const gpu = @import("gpu");
const backend = @import("backend");
const nn = @import("nn.zig");

const m = 37;
const n = 29;
const k = 23;
const lda = 31;
const ldb = 41;
const ldc = 47;

fn fillRandom(prng: *std.Random.DefaultPrng, data: []f32) void {
    const rnd = prng.random();
    for (data) |*x| x.* = rnd.float(f32) * 2 - 1;
}

fn expectPaddingIntact(out: []const f32) !void {
    for (0..m) |i| {
        for (n..ldc) |j| try std.testing.expectEqual(@as(f32, -12345), out[i * ldc + j]);
    }
}

fn referenceNT(r: []f32, a: []const f32, b: []const f32, alpha: f32) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| sum += a[i * lda + p] * b[j * ldb + p];
            r[i * ldc + j] = alpha * sum;
        }
    }
}

fn referenceNN(r: []f32, a: []const f32, b: []const f32) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| sum += a[i * lda + p] * b[p * ldb + j];
            r[i * ldc + j] = sum;
        }
    }
}

test "hardware gemmNT/gemmNN match a CPU reference with padded strides" {
    const be = gpu.context() orelse return error.SkipZigTest;

    var prng = std.Random.DefaultPrng.init(0x5eed);
    var a: [lda * m]f32 = undefined;
    var bnt: [ldb * n]f32 = undefined;
    var bnn: [ldb * k]f32 = undefined;
    var out: [ldc * m]f32 = undefined;
    var ref: [ldc * m]f32 = undefined;
    fillRandom(&prng, &a);
    fillRandom(&prng, &bnt);
    fillRandom(&prng, &bnn);

    @memset(&out, -12345);
    const args_nt: backend.Gemm = .{ .m = m, .n = n, .k = k, .alpha = 0.7, .a = &a, .lda = lda, .b = &bnt, .ldb = ldb, .out = &out, .ldc = ldc };
    // A device that initialized but is not usable reports failure; treat that as "skip".
    if (!be.sgemmNT(args_nt, null)) return error.SkipZigTest;
    referenceNT(&ref, &a, &bnt, 0.7);
    for (0..m) |i| {
        for (0..n) |j| try std.testing.expectApproxEqAbs(ref[i * ldc + j], out[i * ldc + j], 1e-4);
    }
    try expectPaddingIntact(&out);

    // Cached-weight path used by `nn.linear`.
    var weight_data: [n * k]f32 = undefined;
    for (0..n) |i| @memcpy(weight_data[i * k ..][0..k], bnt[i * ldb ..][0..k]);
    const handle = be.registerWeight(&weight_data);
    if (handle == null) return error.SkipZigTest;
    defer be.releaseWeight(handle);
    @memset(&out, -12345);
    if (!be.sgemmNT(.{ .m = m, .n = n, .k = k, .alpha = 0.7, .a = &a, .lda = lda, .b = &weight_data, .ldb = k, .out = &out, .ldc = ldc }, handle)) return error.SkipZigTest;
    referenceNT(&ref, &a, &bnt, 0.7);
    for (0..m) |i| {
        for (0..n) |j| try std.testing.expectApproxEqAbs(ref[i * ldc + j], out[i * ldc + j], 1e-4);
    }
    try expectPaddingIntact(&out);

    // Activation NN path (attention output).
    @memset(&out, -12345);
    if (!be.sgemmNN(.{ .m = m, .n = n, .k = k, .a = &a, .lda = lda, .b = &bnn, .ldb = ldb, .out = &out, .ldc = ldc })) return error.SkipZigTest;
    referenceNN(&ref, &a, &bnn);
    for (0..m) |i| {
        for (0..n) |j| try std.testing.expectApproxEqAbs(ref[i * ldc + j], out[i * ldc + j], 1e-4);
    }
    try expectPaddingIntact(&out);
}

// ---------------------------------------------------------------------------------------------
// Mock backend: no device involved.

const Mock = struct {
    nt_calls: usize = 0,
    nn_calls: usize = 0,
    reg_calls: usize = 0,
    rel_calls: usize = 0,
};

fn mockRegister(ptr: *anyopaque, data: []const f32) ?*anyopaque {
    _ = data;
    const mock: *Mock = @ptrCast(@alignCast(ptr));
    mock.reg_calls += 1;
    return @ptrFromInt(1);
}

fn mockRelease(ptr: *anyopaque, handle: ?*anyopaque) void {
    _ = handle;
    const mock: *Mock = @ptrCast(@alignCast(ptr));
    mock.rel_calls += 1;
}

fn mockNT(ptr: *anyopaque, args: backend.Gemm, weight: ?*anyopaque) bool {
    const mock: *Mock = @ptrCast(@alignCast(ptr));
    mock.nt_calls += 1;
    const value: f32 = if (weight != null) 3.0 else 2.0;
    for (0..args.m) |i| for (0..args.n) |j| {
        args.out[i * args.ldc + j] = value;
    };
    return true;
}

fn mockNN(ptr: *anyopaque, args: backend.Gemm) bool {
    const mock: *Mock = @ptrCast(@alignCast(ptr));
    mock.nn_calls += 1;
    for (0..args.m) |i| for (0..args.n) |j| {
        args.out[i * args.ldc + j] = 4.0;
    };
    return true;
}

const mock_vtable: backend.Backend.VTable = .{
    .register_weight = mockRegister,
    .release_weight = mockRelease,
    .sgemm_nt = mockNT,
    .sgemm_nn = mockNN,
    .linear_min_ops = 1,
    .activation_min_ops = 1,
};

const mock_vtable_high: backend.Backend.VTable = .{
    .register_weight = mockRegister,
    .release_weight = mockRelease,
    .sgemm_nt = mockNT,
    .sgemm_nn = mockNN,
    .linear_min_ops = 1 << 40,
    .activation_min_ops = 1 << 40,
};

test "nn dispatch uses the backend above the threshold and the CPU below it" {
    var mock = Mock{};
    const be: nn.Backend = .{ .ptr = &mock, .vtable = &mock_vtable };
    const io = std.Io.Threaded.global_single_threaded.io();

    var wdata: [4 * 3]f32 = undefined;
    for (&wdata, 0..) |*x, i| x.* = @floatFromInt(i + 1);
    var x: [2 * 3]f32 = .{ 1, 0, 0, 0, 1, 0 };
    var y: [2 * 4]f32 = .{ -1, -1, -1, -1, -1, -1, -1, -1 };
    var bias: [4]f32 = .{ 1, 0, 0, 0 };

    var w = nn.Weight.initWith(be, &wdata);
    try std.testing.expect(mock.reg_calls == 1);
    nn.linear(io, &y, &x, 2, w, 4, 3, &bias);
    try std.testing.expectEqual(@as(usize, 1), mock.nt_calls);
    // Mock writes 3, then `linear` adds the bias.
    try std.testing.expectEqual(@as(f32, 4), y[0]);
    try std.testing.expectEqual(@as(f32, 3), y[1]);

    // Below the threshold the CPU kernel runs and the backend is untouched.
    const be_high: nn.Backend = .{ .ptr = &mock, .vtable = &mock_vtable_high };
    var w_high = nn.Weight.initWith(be_high, &wdata);
    nn.linear(io, &y, &x, 2, w_high, 4, 3, null);
    try std.testing.expectEqual(@as(usize, 1), mock.nt_calls);
    // Row 0 of x = [1,0,0] -> C[0][j] = w[j][0] = [1,4,7,10].
    try std.testing.expectEqual(@as(f32, 1), y[0]);
    try std.testing.expectEqual(@as(f32, 10), y[3]);

    w.deinit();
    w_high.deinit();
    try std.testing.expectEqual(@as(usize, 2), mock.rel_calls);
}

test "nn.gemmNT/gemmNN dispatch to the backend" {
    var mock = Mock{};
    const be: nn.Backend = .{ .ptr = &mock, .vtable = &mock_vtable };
    var a: [1 * 1]f32 = .{1};
    var b: [1 * 1]f32 = .{1};
    var out: [2 * 2]f32 = .{ 0, 0, 0, 0 };
    nn.gemmNT(be, &out, 2, &a, 1, &b, 1, 1, 1, 1, 1.0);
    try std.testing.expectEqual(@as(usize, 1), mock.nt_calls);
    try std.testing.expectEqual(@as(f32, 2), out[0]);
    nn.gemmNN(be, &out, 2, &a, 1, &b, 1, 1, 1, 1);
    try std.testing.expectEqual(@as(usize, 1), mock.nn_calls);
    try std.testing.expectEqual(@as(f32, 4), out[0]);
}
