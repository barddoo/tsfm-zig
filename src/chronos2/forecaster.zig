//! Chronos-2 inference: instance normalization, patching, time encodings, sequence assembly,
//! the encoder pass and the inverse scaling of the quantile head.

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const model_mod = @import("model.zig");

const Allocator = mem.Allocator;
const Model = model_mod.Model;

/// Forecast `horizon` steps after `inputs` `(b, t)` (NaN = missing). Returns `(b, horizon, Q)`
/// quantiles in the model's own quantile order. Series are independent (v1 univariate path).
pub fn forecast(
    allocator: Allocator,
    io: Io,
    model: *const Model,
    inputs: []const f32,
    b: usize,
    t_in: usize,
    horizon: usize,
) ![]f32 {
    const c = model.config;
    const ps: usize = c.chronos_config.input_patch_size;
    const q: usize = c.numQuantiles();
    if (b == 0 or t_in == 0) return error.EmptyInput;
    if (horizon == 0 or horizon > c.predictionLength()) return error.HorizonTooLong;

    // Truncate the context to the model's window (the most recent steps).
    const t = @min(t_in, @as(usize, c.chronos_config.context_length));
    const start = t_in - t;
    const pad = (ps - t % ps) % ps;
    const padded = t + pad;
    const n_ctx = padded / ps;
    const n_out = (horizon + ps - 1) / ps;

    const out = try allocator.alloc(f32, b * horizon * q);
    errdefer allocator.free(out);

    const seq = n_ctx + 1 + n_out;
    const chunk = chunkSize(b, seq, c.d_model);
    var s0: usize = 0;
    while (s0 < b) {
        const s1 = @min(b, s0 + chunk);
        try runChunk(allocator, io, model, out, inputs, s0, s1, t_in, start, t, pad, n_ctx, n_out, horizon);
        s0 = s1;
    }
    return out;
}

/// Bound the encoder's scratch memory while keeping linear layers reasonably wide.
fn chunkSize(b: usize, seq: usize, d: usize) usize {
    const budget: usize = 16 * 1024 * 1024; // floats, roughly 64 MiB
    const per_series = @max(1, seq * d * 24);
    return @min(b, @max(1, budget / per_series));
}

fn runChunk(
    allocator: Allocator,
    io: Io,
    model: *const Model,
    out: []f32,
    inputs: []const f32,
    s0: usize,
    s1: usize,
    stride: usize,
    start: usize,
    t: usize,
    pad: usize,
    n_ctx: usize,
    n_out: usize,
    horizon: usize,
) !void {
    const c = model.config;
    const d: usize = c.d_model;
    const ps: usize = c.chronos_config.input_patch_size;
    const q: usize = c.numQuantiles();
    const seq = n_ctx + 1 + n_out;
    const nb = s1 - s0;
    const tscale: f32 = @floatCast(c.chronos_config.time_encoding_scale);
    const arcsinh = c.chronos_config.use_arcsinh;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const loc = try arena.alloc(f32, nb);
    const scale = try arena.alloc(f32, nb);
    const feats = try arena.alloc(f32, nb * n_ctx * 3 * ps);
    const fut_feats = try arena.alloc(f32, nb * n_out * 3 * ps);
    const ctx_valid = try arena.alloc(bool, nb * n_ctx);

    for (0..nb) |li| {
        const s = s0 + li;
        const row = inputs[s * stride + start ..][0..t];

        // Instance norm over observed values only; NaN -> 0 for loc and NaN -> 1 for scale.
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
            if (sc == 0) sc = 1e-5; // InstanceNorm eps
        }
        loc[li] = lo;
        scale[li] = sc;

        // Patch features `[time_enc, values, mask]`, left-padded with missing steps.
        for (0..n_ctx) |p| {
            var any = false;
            const base = (li * n_ctx + p) * 3 * ps;
            for (0..ps) |i| {
                const j = p * ps + i;
                const src: isize = @as(isize, @intCast(j)) - @as(isize, @intCast(pad));
                var obs = false;
                var val: f32 = 0;
                if (src >= 0 and src < @as(isize, @intCast(t))) {
                    const xv = row[@intCast(src)];
                    if (!std.math.isNan(xv)) {
                        obs = true;
                        val = (xv - lo) / sc;
                        if (arcsinh) val = std.math.asinh(val);
                    }
                }
                feats[base + i] = (@as(f32, @floatFromInt(j)) - @as(f32, @floatFromInt(n_ctx * ps))) / tscale;
                feats[base + ps + i] = val;
                feats[base + 2 * ps + i] = if (obs) 1 else 0;
                any = any or obs;
            }
            ctx_valid[li * n_ctx + p] = any;
        }

        // Future patches carry only their time encoding; v1 sends no known-future covariates.
        for (0..n_out) |p| {
            const base = (li * n_out + p) * 3 * ps;
            for (0..ps) |i| {
                const j = p * ps + i;
                fut_feats[base + i] = @as(f32, @floatFromInt(j)) / tscale;
                fut_feats[base + ps + i] = 0;
                fut_feats[base + 2 * ps + i] = 0;
            }
        }
    }

    const emb = try arena.alloc(f32, nb * n_ctx * d);
    try model.embed(io, arena, emb, feats, nb * n_ctx);
    const fut_emb = try arena.alloc(f32, nb * n_out * d);
    try model.embed(io, arena, fut_emb, fut_feats, nb * n_out);

    // Sequence: context patches, [REG], future patches.
    const x = try arena.alloc(f32, nb * seq * d);
    const key_valid = try arena.alloc(bool, nb * seq);
    for (0..nb) |li| {
        const xb = li * seq * d;
        @memcpy(x[xb..][0 .. n_ctx * d], emb[li * n_ctx * d ..][0 .. n_ctx * d]);
        @memcpy(x[xb + n_ctx * d ..][0..d], model.reg);
        @memcpy(x[xb + (n_ctx + 1) * d ..][0 .. n_out * d], fut_emb[li * n_out * d ..][0 .. n_out * d]);

        @memcpy(key_valid[li * seq ..][0..n_ctx], ctx_valid[li * n_ctx ..][0..n_ctx]);
        key_valid[li * seq + n_ctx] = true;
        @memset(key_valid[li * seq + n_ctx + 1 ..][0..n_out], true);
    }

    try model.encode(io, arena, x, key_valid, nb, n_ctx, n_out);

    const hidden = try arena.alloc(f32, nb * n_out * d);
    for (0..nb) |li| {
        const src = (li * seq + n_ctx + 1) * d;
        @memcpy(hidden[li * n_out * d ..][0 .. n_out * d], x[src..][0 .. n_out * d]);
    }
    const raw = try arena.alloc(f32, nb * n_out * q * ps);
    try model.outputHead(io, arena, raw, hidden, nb * n_out);

    for (0..nb) |li| {
        const s = s0 + li;
        const lo = loc[li];
        const sc = scale[li];
        for (0..n_out) |p| {
            const raw_base = (li * n_out + p) * q * ps;
            for (0..ps) |i| {
                const step = p * ps + i;
                if (step >= horizon) continue;
                for (0..q) |qi| {
                    var v = raw[raw_base + qi * ps + i];
                    if (arcsinh) v = std.math.sinh(v);
                    v = v * sc + lo;
                    out[(s * horizon + step) * q + qi] = v;
                }
            }
        }
    }
}
