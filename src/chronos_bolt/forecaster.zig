//! Chronos-Bolt forecaster: instance normalization, patching, one encoder/decoder pass per block
//! of `prediction_length` steps, and the reference's long-horizon heuristic.
//!
//! Horizons beyond one block feed every predicted quantile back as its own context (so each series
//! is run `num_quantiles` times), then reduce the `num_quantiles^2` resulting samples per step to
//! `num_quantiles` empirical quantiles with linear interpolation, as `ChronosBoltPipeline.predict`
//! does.

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const model_mod = @import("model.zig");

const Allocator = mem.Allocator;
const Model = model_mod.Model;

/// `InstanceNorm`'s replacement for a zero scale.
const norm_eps: f32 = 1e-5;

/// Forecast `horizon` steps after `inputs` `(b, t_in)` (NaN = missing). Returns `(b, horizon, Q)`
/// quantiles in the model's own quantile order. Series are independent.
pub fn forecast(allocator: Allocator, io: Io, model: *const Model, inputs: []const f32, b: usize, t_in: usize, horizon: usize) ![]f32 {
    const c = model.config;
    const q = c.numQuantiles();
    const P: usize = c.chronos_config.prediction_length;
    const max_ctx: usize = c.chronos_config.context_length;
    if (b == 0 or t_in == 0 or horizon == 0) return error.EmptyInput;
    std.debug.assert(inputs.len >= b * t_in);

    const out = try allocator.alloc(f32, b * horizon * q);
    errdefer allocator.free(out);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // First block: the most recent `context_length` steps of every series.
    const len0 = @min(t_in, max_ctx);
    var pred = try arena.alloc(f32, b * q * P); // (b, q, P)
    try predictBlock(allocator, io, model, pred, inputs, t_in, t_in - len0, len0, b);
    storeBlock(out, pred, b, q, P, horizon, 0);
    var done: usize = @min(P, horizon);
    if (done >= horizon) return out;

    // Later blocks: each quantile path becomes its own context row.
    var len = len0;
    var cur = try arena.alloc(f32, b * q * len);
    for (0..b) |s| for (0..q) |qi| @memcpy(cur[(s * q + qi) * len ..][0..len], inputs[s * t_in + t_in - len0 ..][0..len0]);
    const levels = try arena.alloc(f32, q);
    for (levels, c.chronos_config.quantiles) |*l, v| l.* = @floatCast(v);
    const samples = try arena.alloc(f32, q * q);
    const raw = try arena.alloc(f32, b * q * q * P); // (b * q, q, P)

    while (done < horizon) {
        const new_len = @min(len + P, max_ctx);
        const next = try arena.alloc(f32, b * q * new_len);
        const skip = len + P - new_len; // oldest steps dropped from `cur ++ pred`
        for (0..b * q) |row| {
            for (0..new_len) |i| {
                const j = skip + i;
                next[row * new_len + i] = if (j < len) cur[row * len + j] else pred[row * P + (j - len)];
            }
        }
        cur = next;
        len = new_len;

        try predictBlock(allocator, io, model, raw, cur, len, 0, len, b * q);
        // (b * q, q, P) -> (b, q * q, P) samples per step -> q empirical quantiles.
        for (0..b) |s| {
            for (0..P) |p| {
                for (0..q * q) |k| samples[k] = raw[(s * q * q + k) * P + p];
                mem.sort(f32, samples, {}, std.sort.asc(f32));
                for (0..q) |qi| pred[(s * q + qi) * P + p] = quantileLinear(samples, levels[qi]);
            }
        }
        storeBlock(out, pred, b, q, P, horizon, done);
        done += P;
    }
    return out;
}

/// `torch.quantile` (linear interpolation) of an ascending-sorted slice.
fn quantileLinear(sorted: []const f32, level: f32) f32 {
    const rank: f32 = level * @as(f32, @floatFromInt(sorted.len - 1));
    const lo_f = @floor(rank);
    const lo: usize = @intFromFloat(lo_f);
    const hi: usize = @min(@as(usize, @intFromFloat(@ceil(rank))), sorted.len - 1);
    const w = rank - lo_f;
    const a = sorted[lo];
    const bb = sorted[hi];
    // `lerp`: the weight < 0.5 form from the start, otherwise from the end (as ATen does).
    return if (w < 0.5) a + w * (bb - a) else bb - (bb - a) * (1 - w);
}

/// Write block `block_start / P` of `pred` `(b, q, P)` into `out` `(b, horizon, q)`.
fn storeBlock(out: []f32, pred: []const f32, b: usize, q: usize, P: usize, horizon: usize, block_start: usize) void {
    for (0..b) |s| {
        for (0..P) |p| {
            const step = block_start + p;
            if (step >= horizon) break;
            for (0..q) |qi| out[(s * horizon + step) * q + qi] = pred[(s * q + qi) * P + p];
        }
    }
}

/// Bound the pass's scratch memory while keeping linear layers reasonably wide.
fn chunkSize(n: usize, seq: usize, d: usize, ff: usize) usize {
    const budget: usize = 16 * 1024 * 1024; // floats, roughly 64 MiB
    const per_series = @max(1, seq * (d * 12 + ff));
    return @min(n, @max(1, budget / per_series));
}

/// One encoder/decoder pass for `n` series. Row `s` of the context is
/// `ctx[s * stride + offset ..][0..len]`; `out` is `(n, q, P)`.
pub fn predictBlock(allocator: Allocator, io: Io, model: *const Model, out: []f32, ctx: []const f32, stride: usize, offset: usize, len: usize, n: usize) !void {
    const c = model.config;
    const ps: usize = c.chronos_config.input_patch_size;
    const n_patch = (len + ps - 1) / ps;
    const seq = n_patch + @intFromBool(c.chronos_config.use_reg_token);
    const chunk = chunkSize(n, seq, c.d_model, c.d_ff);
    var s0: usize = 0;
    while (s0 < n) {
        const s1 = @min(n, s0 + chunk);
        try runChunk(allocator, io, model, out, ctx, stride, offset, len, s0, s1);
        s0 = s1;
    }
}

fn runChunk(allocator: Allocator, io: Io, model: *const Model, out: []f32, ctx: []const f32, stride: usize, offset: usize, len: usize, s0: usize, s1: usize) !void {
    const c = model.config;
    const d: usize = c.d_model;
    const ps: usize = c.chronos_config.input_patch_size;
    const q = c.numQuantiles();
    const P: usize = c.chronos_config.prediction_length;
    const use_reg = c.chronos_config.use_reg_token;
    const pad = (ps - len % ps) % ps;
    const n_patch = (len + pad) / ps;
    const seq = n_patch + @intFromBool(use_reg);
    const nb = s1 - s0;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const loc = try arena.alloc(f32, nb);
    const scale = try arena.alloc(f32, nb);
    const feats = try arena.alloc(f32, nb * n_patch * 2 * ps);
    const key_valid = try arena.alloc(bool, nb * seq);

    for (0..nb) |li| {
        const row = ctx[(s0 + li) * stride + offset ..][0..len];

        // InstanceNorm over observed values only; NaN -> 0 for loc and NaN -> 1 for scale.
        var count: f64 = 0;
        var sum: f64 = 0;
        for (row) |xv| if (!std.math.isNan(xv)) {
            count += 1;
            sum += xv;
        };
        var lo: f32 = 0;
        var sc: f32 = 1;
        if (count > 0) {
            lo = @floatCast(sum / count);
            var ss: f64 = 0;
            for (row) |xv| if (!std.math.isNan(xv)) {
                const dv: f32 = xv - lo;
                ss += @as(f64, dv) * @as(f64, dv);
            };
            sc = @floatCast(@sqrt(ss / count));
            if (sc == 0) sc = norm_eps;
        }
        loc[li] = lo;
        scale[li] = sc;

        // Patch features `[values, mask]`, left-padded with missing steps; unobserved steps are 0.
        for (0..n_patch) |p| {
            var any = false;
            const base = (li * n_patch + p) * 2 * ps;
            for (0..ps) |i| {
                const src: isize = @as(isize, @intCast(p * ps + i)) - @as(isize, @intCast(pad));
                var obs = false;
                var val: f32 = 0;
                if (src >= 0) {
                    const xv = row[@intCast(src)];
                    if (!std.math.isNan(xv)) {
                        obs = true;
                        val = (xv - lo) / sc;
                    }
                }
                feats[base + i] = val;
                feats[base + ps + i] = if (obs) 1 else 0;
                any = any or obs;
            }
            key_valid[li * seq + p] = any;
        }
        if (use_reg) key_valid[li * seq + n_patch] = true;
    }

    const emb = try arena.alloc(f32, nb * n_patch * d);
    try model.embed(io, arena, emb, feats, nb * n_patch);

    // Sequence: context patches, then [REG].
    const x = try arena.alloc(f32, nb * seq * d);
    for (0..nb) |li| {
        @memcpy(x[li * seq * d ..][0 .. n_patch * d], emb[li * n_patch * d ..][0 .. n_patch * d]);
        if (use_reg) @memcpy(x[(li * seq + n_patch) * d ..][0..d], model.regEmbedding());
    }

    try model.encode(io, arena, x, key_valid, nb, seq);
    const hidden = try arena.alloc(f32, nb * d);
    try model.decode(io, arena, hidden, x, key_valid, nb, seq);
    const raw = try arena.alloc(f32, nb * q * P);
    try model.outputHead(io, arena, raw, hidden, nb);

    // Inverse instance norm.
    for (0..nb) |li| {
        const dst = out[(s0 + li) * q * P ..][0 .. q * P];
        for (dst, raw[li * q * P ..][0 .. q * P]) |*o, v| o.* = v * scale[li] + loc[li];
    }
}

test "quantileLinear interpolates like torch.quantile" {
    const v = [_]f32{ 1, 2, 4, 8, 16 };
    try std.testing.expectEqual(@as(f32, 1), quantileLinear(&v, 0.0));
    try std.testing.expectEqual(@as(f32, 4), quantileLinear(&v, 0.5));
    try std.testing.expectEqual(@as(f32, 16), quantileLinear(&v, 1.0));
    try std.testing.expectApproxEqAbs(@as(f32, 3), quantileLinear(&v, 0.375), 1e-6); // halfway between 2 and 4
}
