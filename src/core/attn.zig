//! Self-attention over one head, specialized at compile time by its mask.
//!
//! `Mask` is a comptime parameter, so each variant compiles to its own loop with no mask tests
//! left in it. Callers that only know the variant at run time use `selfAttendRuntime`, which
//! unrolls one call per variant.

const std = @import("std");
const nn = @import("nn.zig");

pub const Mask = enum {
    /// Every query reads every key.
    full,
    /// Bidirectional; invalid keys get `-floatMax` added to their score (T5-style additive mask).
    keys_additive,
    /// Query `i` reads keys `0..=i`; all keys valid.
    causal,
    /// Causal over valid keys, with the diagonal always open (invalid keys set to -inf).
    causal_keys,
    /// Bidirectional over valid keys, with the diagonal always open (invalid keys set to -inf).
    full_keys,

    /// The variant for a causality flag and whether every key is valid. `additive` selects the
    /// T5-style additive mask for the bidirectional, partially-valid case.
    pub fn pick(causal: bool, all_keys_valid: bool, additive: bool) Mask {
        if (all_keys_valid) return if (causal) .causal else .full;
        if (causal) return .causal_keys;
        return if (additive) .keys_additive else .full_keys;
    }
};

/// Strides (in floats) between consecutive tokens of the q, k, v and output buffers.
pub const Strides = struct { q: usize, k: usize, v: usize, o: usize };

/// Mask and softmax the `n x n` score matrix in place (row stride `n`).
pub inline fn maskedSoftmax(comptime mask: Mask, scores: []f32, n: usize, key_valid: []const bool) void {
    std.debug.assert(scores.len >= n * n);
    if (mask != .full and mask != .causal) std.debug.assert(key_valid.len >= n);
    switch (mask) {
        .full => for (0..n) |i| nn.softmax(scores[i * n ..][0..n]),
        .keys_additive => {
            const mask_min: f32 = -std.math.floatMax(f32);
            for (0..n) |i| {
                const row = scores[i * n ..][0..n];
                for (0..n) |j| if (!key_valid[j]) {
                    row[j] += mask_min;
                };
                nn.softmax(row);
            }
        },
        .causal => for (0..n) |i| {
            // Only a prefix of the row is scored; the tail stays zero.
            nn.softmax(scores[i * n ..][0 .. i + 1]);
            @memset(scores[i * n + i + 1 ..][0 .. n - i - 1], 0);
        },
        .causal_keys, .full_keys => {
            const ninf = -std.math.inf(f32);
            for (0..n) |i| {
                const row = scores[i * n ..][0..n];
                for (0..n) |j| {
                    const allowed = j == i or (key_valid[j] and (mask == .full_keys or j <= i));
                    if (!allowed) row[j] = ninf;
                }
                nn.softmax(row);
            }
        },
    }
}

/// `o = softmax(mask(scale * q k^T + bias)) v` for `n` tokens of one head of width `hd`. `scores`
/// is `n * n` scratch. `key_valid` (length `n`) is read only by the masks that need it. `bias` is
/// an optional additive `n x n` score bias (row stride `n`), applied before the mask.
pub fn selfAttend(
    comptime mask: Mask,
    b: ?nn.Backend,
    scores: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []f32,
    st: Strides,
    n: usize,
    hd: usize,
    scale: f32,
    key_valid: []const bool,
    bias: ?[]const f32,
) void {
    std.debug.assert(scores.len >= n * n);
    std.debug.assert(st.q >= hd and st.k >= hd and st.v >= hd and st.o >= hd);
    std.debug.assert(n == 0 or (q.len >= (n - 1) * st.q + hd and k.len >= (n - 1) * st.k + hd and v.len >= (n - 1) * st.v + hd));
    std.debug.assert(n == 0 or o.len >= (n - 1) * st.o + hd);
    std.debug.assert(bias == null or bias.?.len >= n * n);
    nn.gemmNT(b, scores, n, q, st.q, k, st.k, n, n, hd, scale);
    if (bias) |bv| nn.addInPlace(scores[0 .. n * n], bv[0 .. n * n]);
    maskedSoftmax(mask, scores, n, key_valid);
    nn.gemmNN(b, o, st.o, scores, n, v, st.v, n, hd, n);
}

/// `selfAttend` with the mask chosen at run time: one specialized copy per variant.
pub fn selfAttendRuntime(
    mask: Mask,
    b: ?nn.Backend,
    scores: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []f32,
    st: Strides,
    n: usize,
    hd: usize,
    scale: f32,
    key_valid: []const bool,
    bias: ?[]const f32,
) void {
    switch (mask) {
        inline else => |m| selfAttend(m, b, scores, q, k, v, o, st, n, hd, scale, key_valid, bias),
    }
}

// ---------------------------------------------------------------------------------------------

/// Naive float64 reference: which keys query `i` may read under `mask`.
fn allowedRef(mask: Mask, key_valid: []const bool, i: usize, j: usize) bool {
    return switch (mask) {
        .full => true,
        .keys_additive => key_valid[j],
        .causal => j <= i,
        .causal_keys => j == i or (key_valid[j] and j <= i),
        .full_keys => j == i or key_valid[j],
    };
}

fn referenceAttend(mask: Mask, o: []f64, q: []const f32, k: []const f32, v: []const f32, st: Strides, n: usize, hd: usize, scale: f64, key_valid: []const bool, bias: ?[]const f32) void {
    std.debug.assert(n <= 64);
    var s: [64]f64 = undefined;
    for (0..n) |i| {
        var mx: f64 = -std.math.inf(f64);
        for (0..n) |j| {
            var dot: f64 = 0;
            for (0..hd) |c| dot += @as(f64, q[i * st.q + c]) * @as(f64, k[j * st.k + c]);
            s[j] = dot * scale + if (bias) |b| @as(f64, b[i * n + j]) else 0;
            if (allowedRef(mask, key_valid, i, j)) mx = @max(mx, s[j]);
        }
        var z: f64 = 0;
        for (0..n) |j| {
            s[j] = if (allowedRef(mask, key_valid, i, j)) @exp(s[j] - mx) else 0;
            z += s[j];
        }
        for (0..hd) |c| {
            var acc: f64 = 0;
            for (0..n) |j| acc += s[j] / z * @as(f64, v[j * st.v + c]);
            o[i * hd + c] = acc;
        }
    }
}

test "every mask, with and without a bias, matches a naive reference (strided q/k/v)" {
    const n = 11;
    const hd = 8;
    const ld = 3 * hd + 2; // fused-qkv-like stride with padding
    var prng = std.Random.DefaultPrng.init(0x7a11);
    const rnd = prng.random();
    var qkv: [n * ld]f32 = undefined;
    for (&qkv) |*x| x.* = rnd.float(f32) * 2 - 1;
    var key_valid: [n]bool = undefined;
    for (&key_valid, 0..) |*x, i| x.* = i % 3 != 1; // some invalid keys, including a leading-ish one
    const st: Strides = .{ .q = ld, .k = ld, .v = ld, .o = hd };
    const q = qkv[0..];
    const k = qkv[hd..];
    const v = qkv[2 * hd ..];

    var bias_buf: [n * n]f32 = undefined;
    for (&bias_buf) |*x| x.* = rnd.float(f32) * 4 - 2;

    inline for (comptime std.enums.values(Mask)) |mask| {
        for ([_]?[]const f32{ null, &bias_buf }) |bias| {
            var scores: [n * n]f32 = undefined;
            var got: [n * hd]f32 = undefined;
            var want: [n * hd]f64 = undefined;
            selfAttend(mask, null, &scores, q, k, v, &got, st, n, hd, 0.35, &key_valid, bias);
            referenceAttend(mask, &want, q, k, v, st, n, hd, 0.35, &key_valid, bias);
            for (got, want) |g, w| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(w)), g, 1e-4);

            // The runtime dispatcher must hit the same variant.
            var got_rt: [n * hd]f32 = undefined;
            selfAttendRuntime(mask, null, &scores, q, k, v, &got_rt, st, n, hd, 0.35, &key_valid, bias);
            try std.testing.expectEqualSlices(f32, &got, &got_rt);
        }
    }
}

test "Mask.pick maps causality and key validity to the right variant" {
    try std.testing.expectEqual(Mask.full, Mask.pick(false, true, true));
    try std.testing.expectEqual(Mask.causal, Mask.pick(true, true, false));
    try std.testing.expectEqual(Mask.causal_keys, Mask.pick(true, false, true));
    try std.testing.expectEqual(Mask.keys_additive, Mask.pick(false, false, true));
    try std.testing.expectEqual(Mask.full_keys, Mask.pick(false, false, false));
}
