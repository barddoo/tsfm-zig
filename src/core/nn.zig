//! Dense tensor kernels on flat float32 slices. Large matrix products go to a C BLAS
//! `cblas_sgemm` when one is linked; the builtin kernels (used for `gemmNN` and for the
//! `-Dblas=builtin` build) are register-tiled so a row of B is reused by several rows of A.

const std = @import("std");
const c = @import("c");
const par = @import("par.zig");

/// Whether `gemmNT` (the large products) goes to a C BLAS (Accelerate, OpenBLAS); otherwise the
/// builtin SIMD kernel runs. `gemmNN` (attention output) always uses the builtin kernel.
pub const has_blas = @hasDecl(c, "cblas_sgemm");

/// `C[m, n] = alpha * A[m, k] @ B^T` with B stored `[n, k]`, explicit row strides.
pub fn gemmNT(cc: []f32, ldc: usize, a: []const f32, lda: usize, b: []const f32, ldb: usize, m: usize, n: usize, k: usize, alpha: f32) void {
    if (comptime has_blas) {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans, @intCast(m), @intCast(n), @intCast(k), alpha, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), 0.0, cc.ptr, @intCast(ldc));
        return;
    }
    const kv = k / lanes * lanes;
    // Register tile: rows of A against four rows of B. Wide targets (AVX2) have only 16 vector
    // registers, so they use a 2-row tile to keep all accumulators and operands in registers;
    // NEON (4 lanes) has 32 and can afford 4 rows.
    const row_tile = if (lanes > 4) 2 else 4;
    var i: usize = 0;
    // Without the row tile every row of B is streamed once per row of A, which makes the builtin
    // kernel memory-bound.
    while (i + row_tile <= m) : (i += row_tile) {
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            var acc: [row_tile][4]Vec = undefined;
            inline for (0..row_tile) |u| {
                inline for (0..4) |w| acc[u][w] = @splat(0);
            }
            var p: usize = 0;
            while (p < kv) : (p += lanes) {
                var av: [row_tile]Vec = undefined;
                inline for (0..row_tile) |u| av[u] = a[(i + u) * lda + p ..][0..lanes].*;
                var bv: [4]Vec = undefined;
                inline for (0..4) |w| bv[w] = b[(j + w) * ldb + p ..][0..lanes].*;
                inline for (0..row_tile) |u| {
                    inline for (0..4) |w| acc[u][w] += av[u] * bv[w];
                }
            }
            inline for (0..row_tile) |u| {
                inline for (0..4) |w| {
                    var sum = @reduce(.Add, acc[u][w]);
                    var q = kv;
                    while (q < k) : (q += 1) sum += a[(i + u) * lda + q] * b[(j + w) * ldb + q];
                    cc[(i + u) * ldc + j + w] = alpha * sum;
                }
            }
        }
        // Column remainder: one column against the tiled rows.
        while (j < n) : (j += 1) {
            var acc: [row_tile]Vec = undefined;
            inline for (0..row_tile) |u| acc[u] = @splat(0);
            var p: usize = 0;
            while (p < kv) : (p += lanes) {
                const bv: Vec = b[j * ldb + p ..][0..lanes].*;
                inline for (0..row_tile) |u| {
                    const av: Vec = a[(i + u) * lda + p ..][0..lanes].*;
                    acc[u] += av * bv;
                }
            }
            inline for (0..row_tile) |u| {
                var sum = @reduce(.Add, acc[u]);
                var q = kv;
                while (q < k) : (q += 1) sum += a[(i + u) * lda + q] * b[j * ldb + q];
                cc[(i + u) * ldc + j] = alpha * sum;
            }
        }
    }
    // Row remainder: the original one-row kernel, four B rows at a time.
    while (i < m) : (i += 1) {
        const ar = a[i * lda ..];
        var j: usize = 0;
        while (j + 4 <= n) : (j += 4) {
            var acc: [4]Vec = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
            var p: usize = 0;
            while (p < kv) : (p += lanes) {
                const av: Vec = ar[p..][0..lanes].*;
                inline for (0..4) |u| {
                    const bv: Vec = b[(j + u) * ldb + p ..][0..lanes].*;
                    acc[u] += av * bv;
                }
            }
            inline for (0..4) |u| {
                var sum = @reduce(.Add, acc[u]);
                var q = kv;
                while (q < k) : (q += 1) sum += ar[q] * b[(j + u) * ldb + q];
                cc[i * ldc + j + u] = alpha * sum;
            }
        }
        while (j < n) : (j += 1) {
            var acc: Vec = @splat(0);
            var p: usize = 0;
            while (p < kv) : (p += lanes) {
                const av: Vec = ar[p..][0..lanes].*;
                const bv: Vec = b[j * ldb + p ..][0..lanes].*;
                acc += av * bv;
            }
            var sum = @reduce(.Add, acc);
            var q = kv;
            while (q < k) : (q += 1) sum += ar[q] * b[j * ldb + q];
            cc[i * ldc + j] = alpha * sum;
        }
    }
}

/// `C[m, n] = A[m, k] @ B[k, n]` with explicit row strides.
pub fn gemmNN(cc: []f32, ldc: usize, a: []const f32, lda: usize, b: []const f32, ldb: usize, m: usize, n: usize, k: usize) void {
    if (comptime has_blas) {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans, @intCast(m), @intCast(n), @intCast(k), 1.0, a.ptr, @intCast(lda), b.ptr, @intCast(ldb), 0.0, cc.ptr, @intCast(ldc));
        return;
    }
    const nv = n / lanes * lanes;
    var i: usize = 0;
    // Four rows of A share every load of B.
    while (i + 4 <= m) : (i += 4) {
        const c0 = cc[(i + 0) * ldc ..][0..n];
        const c1 = cc[(i + 1) * ldc ..][0..n];
        const c2 = cc[(i + 2) * ldc ..][0..n];
        const c3 = cc[(i + 3) * ldc ..][0..n];
        @memset(c0, 0);
        @memset(c1, 0);
        @memset(c2, 0);
        @memset(c3, 0);
        for (0..k) |p| {
            const a0 = a[(i + 0) * lda + p];
            const a1 = a[(i + 1) * lda + p];
            const a2 = a[(i + 2) * lda + p];
            const a3 = a[(i + 3) * lda + p];
            if (a0 == 0 and a1 == 0 and a2 == 0 and a3 == 0) continue;
            const v0: Vec = @splat(a0);
            const v1: Vec = @splat(a1);
            const v2: Vec = @splat(a2);
            const v3: Vec = @splat(a3);
            const brow = b[p * ldb ..][0..n];
            var j: usize = 0;
            while (j < nv) : (j += lanes) {
                const bv: Vec = brow[j..][0..lanes].*;
                const x0: Vec = c0[j..][0..lanes].*;
                const x1: Vec = c1[j..][0..lanes].*;
                const x2: Vec = c2[j..][0..lanes].*;
                const x3: Vec = c3[j..][0..lanes].*;
                c0[j..][0..lanes].* = x0 + v0 * bv;
                c1[j..][0..lanes].* = x1 + v1 * bv;
                c2[j..][0..lanes].* = x2 + v2 * bv;
                c3[j..][0..lanes].* = x3 + v3 * bv;
            }
            while (j < n) : (j += 1) {
                c0[j] += a0 * brow[j];
                c1[j] += a1 * brow[j];
                c2[j] += a2 * brow[j];
                c3[j] += a3 * brow[j];
            }
        }
    }
    while (i < m) : (i += 1) {
        const crow = cc[i * ldc ..][0..n];
        @memset(crow, 0);
        for (0..k) |p| {
            const av = a[i * lda + p];
            if (av == 0) continue;
            const avv: Vec = @splat(av);
            const brow = b[p * ldb ..][0..n];
            var j: usize = 0;
            while (j < nv) : (j += lanes) {
                const cv: Vec = crow[j..][0..lanes].*;
                const bv: Vec = brow[j..][0..lanes].*;
                crow[j..][0..lanes].* = cv + avv * bv;
            }
            while (j < n) : (j += 1) crow[j] += av * brow[j];
        }
    }
}

const LinearCtx = struct {
    y: []f32,
    x: []const f32,
    rows: usize,
    w: []const f32,
    out: usize,
    in: usize,
    bias: ?[]const f32,
};

/// Output columns `[j0, j1)` of a linear layer; the weight rows of those columns are contiguous.
fn linearCols(ctx: *const LinearCtx, j0: usize, j1: usize) void {
    gemmNT(ctx.y[j0..], ctx.out, ctx.x, ctx.in, ctx.w[j0 * ctx.in ..], ctx.in, ctx.rows, j1 - j0, ctx.in, 1.0);
    if (ctx.bias) |b| {
        for (0..ctx.rows) |r| addInPlace(ctx.y[r * ctx.out + j0 ..][0 .. j1 - j0], b[j0..j1]);
    }
}

const col_block = 16;

fn linearBlocks(ctx: *const LinearCtx, b0: usize, b1: usize) void {
    const nblocks = (ctx.out + col_block - 1) / col_block;
    linearCols(ctx, b0 * col_block, if (b1 == nblocks) ctx.out else b1 * col_block);
}

/// `y[rows, out] = x[rows, in] @ w[out, in]^T`, plus `bias[out]` when given. Large products are
/// split by output columns across the thread pool.
pub fn linear(io: std.Io, y: []f32, x: []const f32, rows: usize, w: []const f32, out: usize, in: usize, bias: ?[]const f32) void {
    std.debug.assert(y.len >= rows * out and x.len >= rows * in and w.len == out * in);
    const ctx: LinearCtx = .{ .y = y, .x = x, .rows = rows, .w = w, .out = out, .in = in, .bias = bias };
    // Always use the same column blocking, even on one thread: the C BLAS may pick a different
    // kernel (and rounding) for a different `n`, which would make the output depend on the
    // thread count. Threads only decide how many blocks run concurrently.
    if (rows * out * in >= (1 << 20) and out >= 2 * col_block) {
        par.forRange(io, (out + col_block - 1) / col_block, &ctx, linearBlocks);
    } else {
        linearCols(&ctx, 0, out);
    }
}

/// SIMD lane count for f32 on the build target.
pub const lanes = std.simd.suggestVectorLength(f32) orelse 4;
const Vec = @Vector(lanes, f32);
const IVec = @Vector(lanes, i32);

/// `dst[i] += src[i]`. Residual adds are on the hot path of every layer, so they get an explicit
/// vector loop rather than hoping the optimizer proves that `dst` and `src` cannot alias.
pub fn addInPlace(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len >= src.len);
    var i: usize = 0;
    while (i + lanes <= src.len) : (i += lanes) {
        const a: Vec = dst[i..][0..lanes].*;
        const b: Vec = src[i..][0..lanes].*;
        dst[i..][0..lanes].* = a + b;
    }
    while (i < src.len) : (i += 1) dst[i] += src[i];
}

/// `dst[i] += scalar * src[i]`.
pub fn mulAddInPlace(dst: []f32, src: []const f32, scalar: f32) void {
    std.debug.assert(dst.len >= src.len);
    const s: Vec = @splat(scalar);
    var i: usize = 0;
    while (i + lanes <= src.len) : (i += lanes) {
        const a: Vec = dst[i..][0..lanes].*;
        const b: Vec = src[i..][0..lanes].*;
        dst[i..][0..lanes].* = a + s * b;
    }
    while (i < src.len) : (i += 1) dst[i] += scalar * src[i];
}

/// Contiguous dot product. Head slices are stored contiguously, so attention reduces a few
/// vectors instead of one element at a time.
pub fn dot(a: []const f32, b: []const f32) f32 {
    const n = @min(a.len, b.len);
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const av: Vec = a[i..][0..lanes].*;
        const bv: Vec = b[i..][0..lanes].*;
        acc += av * bv;
    }
    var sum = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// `exp` on a vector: range reduction `x = n ln2 + r` and a degree-6 polynomial for `e^r`
/// (Cephes `expf`, about 1 ulp). The input is clamped to the finite float32 range of `exp`.
inline fn expVec(x_in: Vec) Vec {
    const x = @min(@max(x_in, @as(Vec, @splat(-87.33654))), @as(Vec, @splat(88.72284)));
    const fx = @round(x * @as(Vec, @splat(1.44269504088896341)));
    var r = x - fx * @as(Vec, @splat(0.693359375));
    r = r - fx * @as(Vec, @splat(-2.12194440e-4));
    var y: Vec = @splat(1.9875691500e-4);
    y = y * r + @as(Vec, @splat(1.3981999507e-3));
    y = y * r + @as(Vec, @splat(8.3334519073e-3));
    y = y * r + @as(Vec, @splat(4.1665795894e-2));
    y = y * r + @as(Vec, @splat(1.6666665459e-1));
    y = y * r + @as(Vec, @splat(5.0000001201e-1));
    y = y * r * r + r + @as(Vec, @splat(1.0));
    const n: IVec = @intFromFloat(fx);
    const pow2: Vec = @bitCast((n + @as(IVec, @splat(127))) << @as(@Vector(lanes, u5), @splat(23)));
    return y * pow2;
}

const NormCtx = struct { out: []f32, x: []const f32, dim: usize, scale: []const f32, eps: f32 };

fn rmsNormRows(ctx: *const NormCtx, r0: usize, r1: usize) void {
    rmsNormSerial(ctx.out, ctx.x, r0, r1, ctx.dim, ctx.scale, ctx.eps);
}

pub fn rmsNorm(io: std.Io, out: []f32, x: []const f32, rows: usize, dim: usize, scale: []const f32, eps: f32) void {
    const ctx: NormCtx = .{ .out = out, .x = x, .dim = dim, .scale = scale, .eps = eps };
    par.forRangeMin(io, rows, 64, &ctx, rmsNormRows);
}

fn rmsNormSerial(out: []f32, x: []const f32, r0: usize, r1: usize, dim: usize, scale: []const f32, eps: f32) void {
    for (r0..r1) |r| {
        const xr = x[r * dim ..][0..dim];
        var acc: Vec = @splat(0);
        var i: usize = 0;
        while (i + lanes <= dim) : (i += lanes) {
            const v: Vec = xr[i..][0..lanes].*;
            acc += v * v;
        }
        var ss: f32 = @reduce(.Add, acc);
        while (i < dim) : (i += 1) ss += xr[i] * xr[i];
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dim)) + eps);
        const o = out[r * dim ..][0..dim];
        const invv: Vec = @splat(inv);
        i = 0;
        while (i + lanes <= dim) : (i += lanes) {
            const v: Vec = xr[i..][0..lanes].*;
            const sc: Vec = scale[i..][0..lanes].*;
            o[i..][0..lanes].* = v * invv * sc;
        }
        while (i < dim) : (i += 1) o[i] = xr[i] * inv * scale[i];
    }
}

/// In-place SiLU (`x * sigmoid(x)`), using the same vector `exp` as `mulSilu`.
pub fn siluInPlace(x: []f32) void {
    const one: Vec = @splat(1.0);
    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const v: Vec = x[i..][0..lanes].*;
        x[i..][0..lanes].* = v / (one + expVec(-v));
    }
    while (i < x.len) : (i += 1) x[i] = silu(x[i]);
}

/// `dst[i] = gate[i] * silu(up[i])`.
pub fn mulSilu(dst: []f32, gate: []const f32, up: []const f32) void {
    const one: Vec = @splat(1.0);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        const u: Vec = up[i..][0..lanes].*;
        const g: Vec = gate[i..][0..lanes].*;
        dst[i..][0..lanes].* = g * (u / (one + expVec(-u)));
    }
    while (i < dst.len) : (i += 1) dst[i] = gate[i] * silu(up[i]);
}

/// `dst[i] = x[i] * sigmoid(g[i])`.
pub fn mulSigmoid(dst: []f32, x: []const f32, g: []const f32) void {
    const one: Vec = @splat(1.0);
    var i: usize = 0;
    while (i + lanes <= dst.len) : (i += lanes) {
        const xv: Vec = x[i..][0..lanes].*;
        const gv: Vec = g[i..][0..lanes].*;
        dst[i..][0..lanes].* = xv / (one + expVec(-gv));
    }
    while (i < dst.len) : (i += 1) dst[i] = x[i] * sigmoid(g[i]);
}

/// In-place ReLU.
pub fn relu(x: []f32) void {
    const zero: Vec = @splat(0);
    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const v: Vec = x[i..][0..lanes].*;
        x[i..][0..lanes].* = @max(v, zero);
    }
    while (i < x.len) : (i += 1) x[i] = @max(x[i], 0);
}

/// In-place RoPE over one head in the Llama `rotate_half` layout: dims `i` and `i + half` are
/// paired, and `cos` / `sin` hold `half` angles.
pub fn ropeHalf(x: []f32, cos: []const f32, sin: []const f32, half: usize) void {
    std.debug.assert(x.len >= 2 * half and cos.len >= half and sin.len >= half);
    const lo = x[0..half];
    const hi = x[half..][0..half];
    var i: usize = 0;
    while (i + lanes <= half) : (i += lanes) {
        const a: Vec = lo[i..][0..lanes].*;
        const b: Vec = hi[i..][0..lanes].*;
        const cv: Vec = cos[i..][0..lanes].*;
        const sv: Vec = sin[i..][0..lanes].*;
        lo[i..][0..lanes].* = a * cv - b * sv;
        hi[i..][0..lanes].* = b * cv + a * sv;
    }
    while (i < half) : (i += 1) {
        const x1 = lo[i];
        const x2 = hi[i];
        lo[i] = x1 * cos[i] - x2 * sin[i];
        hi[i] = x2 * cos[i] + x1 * sin[i];
    }
}

/// Softmax in place; entries equal to -inf get exactly zero weight.
pub fn softmax(row: []f32) void {
    const ninf = -std.math.inf(f32);
    const ninfv: Vec = @splat(ninf);
    var mxv: Vec = @splat(ninf);
    var i: usize = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        const v: Vec = row[i..][0..lanes].*;
        mxv = @max(mxv, v);
    }
    var mx: f32 = @reduce(.Max, mxv);
    while (i < row.len) : (i += 1) mx = @max(mx, row[i]);
    const mxsv: Vec = @splat(mx);
    const zero: Vec = @splat(0);
    var sumv: Vec = @splat(0);
    i = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        const v: Vec = row[i..][0..lanes].*;
        const e = @select(f32, v == ninfv, zero, expVec(v - mxsv));
        row[i..][0..lanes].* = e;
        sumv += e;
    }
    var sum: f32 = @reduce(.Add, sumv);
    while (i < row.len) : (i += 1) {
        row[i] = if (row[i] == ninf) 0 else @exp(row[i] - mx);
        sum += row[i];
    }
    const inv: Vec = @splat(1.0 / sum);
    const invs = 1.0 / sum;
    i = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        const v: Vec = row[i..][0..lanes].*;
        row[i..][0..lanes].* = v * inv;
    }
    while (i < row.len) : (i += 1) row[i] *= invs;
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// `torch.nn.functional.softplus` (beta=1, threshold=20).
pub inline fn softplus(x: f32) f32 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

/// Maps +/-inf to the nearest finite value of the vector; an all non-finite vector becomes 0.
/// NaN is left untouched.
pub fn clampNonfinite(v: []f32) void {
    const inf = std.math.inf(f32);
    const infv: Vec = @splat(inf);
    const ninfv: Vec = @splat(-inf);
    const zero: Vec = @splat(0);
    var hi_v: Vec = @splat(-inf);
    var lo_v: Vec = @splat(inf);
    var any_inf = false;
    var i: usize = 0;
    while (i + lanes <= v.len) : (i += lanes) {
        const x: Vec = v[i..][0..lanes].*;
        const finite = @abs(x) < infv; // false for both NaN and +/-inf
        hi_v = @max(hi_v, @select(f32, finite, x, ninfv));
        lo_v = @min(lo_v, @select(f32, finite, x, infv));
        any_inf = any_inf or @reduce(.Or, @abs(x) == infv);
    }
    var hi: f32 = @reduce(.Max, hi_v);
    var lo: f32 = @reduce(.Min, lo_v);
    while (i < v.len) : (i += 1) {
        const x = v[i];
        if (std.math.isFinite(x)) {
            hi = @max(hi, x);
            lo = @min(lo, x);
        } else if (std.math.isInf(x)) any_inf = true;
    }
    if (!any_inf) return;
    const hiv: Vec = @splat(hi);
    const lov: Vec = @splat(lo);
    i = 0;
    while (i + lanes <= v.len) : (i += lanes) {
        const x: Vec = v[i..][0..lanes].*;
        var d = @select(f32, x == infv, hiv, x);
        d = @select(f32, d == ninfv, lov, d);
        d = @select(f32, @abs(d) == infv, zero, d);
        v[i..][0..lanes].* = d;
    }
    while (i < v.len) : (i += 1) {
        if (v[i] == std.math.inf(f32)) v[i] = hi;
        if (v[i] == -std.math.inf(f32)) v[i] = lo;
        if (std.math.isInf(v[i])) v[i] = 0;
    }
}
