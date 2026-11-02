//! The Timestral network (context-only path): causal per-patch normalization, patch
//! embedding, a time/variate transformer over `[Enc ; Dec]` tokens, and the monotone
//! quantile head. Mirrors `timestral.model`, `timestral.layers` and `timestral.masks`.
//!
//! Tensors use the packed layout: `R` rows (the target variates and covariates of every
//! series), `N` patches, `D` channels, stored `[R][N][D]`. Rows of one series are contiguous.

const std = @import("std");
const cfg = @import("config.zig");
const core = @import("core");
const nn = core.nn;
const st = core.safetensors;
const par = core.par;

const Allocator = std.mem.Allocator;

const Linear = struct {
    w: []const f32,
    b: ?[]const f32 = null,
};

const Attn = struct {
    is_time: bool,
    norm: []const f32,
    /// Fused `[q ; k ; v]` projection, rows permuted to `(qkv, head, head_dim)` at load time.
    qkv: []f32,
    out: []const f32,
    gate: ?[]const f32,
};

const MlpAct = struct {
    h: []const f32,
    a: []f32,
    hid: usize,

    /// `gate * silu(up)` for tokens `[first, last)`; the fused projection holds `[gate | up]` per token.
    fn run(self: *const MlpAct, first: usize, last: usize) void {
        for (first..last) |i| {
            const hv = self.h[i * 2 * self.hid ..][0 .. 2 * self.hid];
            nn.mulSilu(self.a[i * self.hid ..][0..self.hid], hv[0..self.hid], hv[self.hid..]);
        }
    }
};

const Layer = struct {
    attns: [2]Attn,
    n_attn: usize,
    mlp_norm: []const f32,
    w12: []const f32,
    w3: []const f32,
};

const ResBlock = struct {
    hidden: Linear,
    output: Linear,
    residual: Linear,
    in: usize,
    hid: usize,
    out: usize,

    fn load(io: std.Io, w: *st.Weights, buf: []u8, prefix: []const u8, in: usize, hid: usize, out: usize, bias: bool) !ResBlock {
        return .{
            .hidden = try loadLinear(io, w, buf, prefix, "hidden_layer", hid, in, bias),
            .output = try loadLinear(io, w, buf, prefix, "output_layer", out, hid, bias),
            .residual = try loadLinear(io, w, buf, prefix, "residual_layer", out, in, bias),
            .in = in,
            .hid = hid,
            .out = out,
        };
    }

    /// `output(silu(hidden(x))) + residual(x)` for `rows` input vectors.
    fn forward(self: ResBlock, io: std.Io, allocator: Allocator, y: []f32, x: []const f32, rows: usize) !void {
        const h = try allocator.alloc(f32, rows * self.hid);
        defer allocator.free(h);
        nn.linear(io, h, x, rows, self.hidden.w, self.hid, self.in, self.hidden.b);
        nn.siluInPlace(h);
        nn.linear(io, y, h, rows, self.output.w, self.out, self.hid, self.output.b);
        const r = try allocator.alloc(f32, rows * self.out);
        defer allocator.free(r);
        nn.linear(io, r, x, rows, self.residual.w, self.out, self.in, self.residual.b);
        nn.addInPlace(y[0 .. rows * self.out], r);
    }
};

fn tensorName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

fn loadLinear(io: std.Io, w: *st.Weights, buf: []u8, prefix: []const u8, name: []const u8, out: usize, in: usize, bias: bool) !Linear {
    const wn = tensorName(buf, "{s}.{s}.weight", .{ prefix, name });
    const weight = try w.get(io, wn, out * in);
    var bv: ?[]const f32 = null;
    if (bias) {
        const bn = tensorName(buf, "{s}.{s}.bias", .{ prefix, name });
        bv = try w.get(io, bn, out);
    }
    return .{ .w = weight, .b = bv };
}

/// The checkpoint stores the fused projection rows as `(qkv, head_dim, heads)`. Reorder them to
/// `(qkv, head, head_dim)` once, at load time: every attention head is then a contiguous slice, so
/// Q/K/V can be read by the vector kernels directly instead of being gathered per token.
fn permuteQkv(allocator: Allocator, src: []const f32, d: usize, nh: usize, hd: usize) ![]f32 {
    std.debug.assert(d == nh * hd);
    const dst = try allocator.alloc(f32, 3 * d * d);
    for (0..3) |sel| {
        for (0..nh) |hh| {
            for (0..hd) |dd| {
                const src_row = sel * d + dd * nh + hh;
                const dst_row = (sel * nh + hh) * hd + dd;
                @memcpy(dst[dst_row * d ..][0..d], src[src_row * d ..][0..d]);
            }
        }
    }
    return dst;
}

/// What a row of the packed layout is. Covariates condition the targets but never read them;
/// only known-future covariates see the forecast horizon.
pub const Role = enum { target, past, known };

/// One decode block's input. Rows are grouped by series: `groups` series of `roles.len` rows
/// each, in the order given by `roles` (the same for every series).
pub const BlockInput = struct {
    ctx: []const f32, // (R, T)
    obs: []const bool, // (R, T)
    t: usize,
    groups: usize,
    roles: []const Role,
    h: usize, // patches to forecast
    /// `(R, h * patch_size)` known-future covariate values and their observation mask.
    known_future: ?[]const f32 = null,
    known_future_obs: ?[]const bool = null,
};

const Geo = struct {
    rows: usize,
    groups: usize,
    vall: usize, // rows per group
    ne: usize, // encoder tokens per row
    l: usize, // context patches
    h: usize, // decoder tokens per row
    roles: []const Role,
    cov: bool, // covariate window path (firewalled variate attention, bidirectional known rows)
    key_valid: []const bool, // (R, ne)
    keys_all_valid: bool, // every key of every row is valid (no mask needed)
    first_obs: []const usize, // per group

    fn s(self: Geo) usize {
        return self.ne + self.h;
    }
    fn role(self: Geo, r: usize) Role {
        return self.roles[r % self.vall];
    }
};

pub const Model = struct {
    config: cfg.Config,
    patch_proj: ResBlock,
    forecast_query: []const f32,
    layers: []Layer,
    out_norm: []const f32,
    head: ResBlock,
    inv_freq: []f32, // head_dim / 2 entries; zero past rope_dim / 2 (identity rotation)

    pub fn load(io: std.Io, allocator: Allocator, config: cfg.Config, w: *st.Weights) !Model {
        const d: usize = config.d_model;
        const ps: usize = config.patch_size;
        const hid: usize = config.mlp_hidden_dim;
        var buf: [256]u8 = undefined;

        const patch_proj = try ResBlock.load(io, w, &buf, "representation.patch_proj", 2 * ps, 4 * d, d, true);
        const forecast_query = try w.get(io, "representation.forecast_query", d);
        const head = try ResBlock.load(io, w, &buf, "head.projection", d, 4 * d, ps * config.numQuantiles(), false);
        const out_norm = try w.get(io, "transformer.out_norm.scale", d);

        const layers = try allocator.alloc(Layer, config.block_schedule.len);
        errdefer allocator.free(layers);
        for (config.block_schedule, 0..) |kind, li| {
            const axes = try cfg.layerAxes(kind);
            var layer: Layer = undefined;
            layer.n_attn = axes.len;
            for (0..axes.len) |ai| {
                const is_time = axes.is_time[ai];
                const gate_kind = if (is_time) config.time_attn_gate else config.variate_attn_gate;
                var a: Attn = undefined;
                a.is_time = is_time;
                a.norm = try w.get(io, tensorName(&buf, "transformer.layers.{d}.attn_norms.{d}.scale", .{ li, ai }), d);
                const nh: usize = config.num_heads;
                const hd: usize = config.headDim();
                a.qkv = try permuteQkv(allocator, try w.get(io, tensorName(&buf, "transformer.layers.{d}.attns.{d}.wQKV.weight", .{ li, ai }), 3 * d * d), d, nh, hd);
                a.out = try w.get(io, tensorName(&buf, "transformer.layers.{d}.attns.{d}.wO.weight", .{ li, ai }), d * d);
                a.gate = if (std.mem.eql(u8, gate_kind, "sdpa_sigmoid"))
                    try w.get(io, tensorName(&buf, "transformer.layers.{d}.attns.{d}._gate.weight", .{ li, ai }), d * d)
                else
                    null;
                layer.attns[ai] = a;
            }
            layer.mlp_norm = try w.get(io, tensorName(&buf, "transformer.layers.{d}.mlp_norm.scale", .{li}), d);
            layer.w12 = try w.get(io, tensorName(&buf, "transformer.layers.{d}.mlp.w12.weight", .{li}), 2 * hid * d);
            layer.w3 = try w.get(io, tensorName(&buf, "transformer.layers.{d}.mlp.w3.weight", .{li}), d * hid);
            layers[li] = layer;
        }

        const hd: usize = config.headDim();
        const rd: usize = config.ropeDim();
        const inv_freq = try allocator.alloc(f32, hd / 2);
        @memset(inv_freq, 0);
        const base: f32 = @floatCast(config.rope_theta);
        for (0..rd / 2) |i| {
            const idx: f32 = @floatFromInt(2 * i);
            inv_freq[i] = 1.0 / std.math.pow(f32, base, idx / @as(f32, @floatFromInt(rd)));
        }

        return .{
            .config = config,
            .patch_proj = patch_proj,
            .forecast_query = forecast_query,
            .layers = layers,
            .out_norm = out_norm,
            .head = head,
            .inv_freq = inv_freq,
        };
    }

    pub fn deinit(self: *Model, allocator: Allocator) void {
        for (self.layers) |layer| {
            for (layer.attns[0..layer.n_attn]) |a| allocator.free(a.qkv);
        }
        allocator.free(self.layers);
        allocator.free(self.inv_freq);
    }

    /// Per-patch causal `(loc, scale)` of packed `(R, T)` rows, each `(R, T / patch_size)`.
    /// Each patch uses the running mean/std (float64 Welford) of everything observed up to
    /// its last step, then both are rounded to float32.
    pub fn patchStats(self: *const Model, io: std.Io, loc: []f32, scale: []f32, data: []const f32, observed: []const bool, rows: usize, t: usize) void {
        const job: StatsJob = .{ .model = self, .loc = loc, .scale = scale, .data = data, .observed = observed, .t = t };
        par.forRangeMin(io, rows, 4, &job, StatsJob.run);
    }

    /// Quantiles `(T, h * patch_size, Q)` in data units for the target rows after the packed
    /// context, `T` being the number of target rows (series by series, in `roles` order).
    pub fn predictBlock(self: *const Model, io: std.Io, allocator: Allocator, out: []f32, in: BlockInput) !void {
        const c = self.config;
        const ps: usize = c.patch_size;
        const d: usize = c.d_model;
        const q: usize = c.numQuantiles();
        const t = in.t;
        const rows = in.ctx.len / t;
        const vall = in.roles.len;
        const h = in.h;
        const l = t / ps;
        const hl = h * ps;

        var cov = false;
        var n_tgt: usize = 0;
        for (in.roles) |role| {
            cov = cov or role != .target;
            if (role == .target) n_tgt += 1;
        }
        if (!cov and in.known_future != null) return error.InvalidInput;
        if (cov and (h < 1 or h > c.block_size)) return error.BlockTooLong;

        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // The window: the context, plus (with covariates) the horizon blanks / known futures.
        const w = if (cov) t + hl else t;
        const n = w / ps;
        var full = in.ctx;
        var full_obs = in.obs;
        if (cov) {
            const f = try arena.alloc(f32, rows * w);
            const fo = try arena.alloc(bool, rows * w);
            for (0..rows) |r| {
                @memcpy(f[r * w ..][0..t], in.ctx[r * t ..][0..t]);
                @memcpy(fo[r * w ..][0..t], in.obs[r * t ..][0..t]);
                if (in.known_future) |kf| {
                    @memcpy(f[r * w + t ..][0..hl], kf[r * hl ..][0..hl]);
                    @memcpy(fo[r * w + t ..][0..hl], in.known_future_obs.?[r * hl ..][0..hl]);
                } else {
                    @memset(f[r * w + t ..][0..hl], 0);
                    @memset(fo[r * w + t ..][0..hl], false);
                }
            }
            full = f;
            full_obs = fo;
        }

        // Causal per-patch normalization.
        const loc = try arena.alloc(f32, rows * n);
        const scale = try arena.alloc(f32, rows * n);
        self.patchStats(io, loc, scale, full, full_obs, rows, w);
        const eps32 = std.math.floatEps(f32);
        const emb_in = try arena.alloc(f32, rows * n * 2 * ps);
        for (0..rows) |r| {
            const known = cov and in.roles[r % vall] == .known;
            // Known-future covariates use whole-row statistics; 0/1 indicator rows skip the asinh.
            var kloc: f32 = 0;
            var kscale: f32 = 1;
            var binary = false;
            if (known) {
                var cnt: f32 = 0;
                var sum: f32 = 0;
                var zero_one = true;
                var any = false;
                for (0..w) |i| if (full_obs[r * w + i]) {
                    const x = full[r * w + i];
                    cnt += 1;
                    sum += x;
                    zero_one = zero_one and (x == 0 or x == 1);
                    any = true;
                };
                kloc = sum / @max(cnt, 1.0);
                var ss: f32 = 0;
                for (0..w) |i| if (full_obs[r * w + i]) {
                    const dv = full[r * w + i] - kloc;
                    ss += dv * dv;
                };
                kscale = @max(@sqrt(ss / @max(cnt, 1.0)), @as(f32, @floatCast(c.minimum_scale)));
                binary = zero_one and any;
                if (binary) {
                    kloc = 0;
                    kscale = 1;
                }
            }
            for (0..n) |p| {
                const lo = if (known) kloc else loc[r * n + p];
                const sc = @max(if (known) kscale else scale[r * n + p], eps32);
                const dst = emb_in[(r * n + p) * 2 * ps ..][0 .. 2 * ps];
                for (0..ps) |i| {
                    const obs = full_obs[r * w + p * ps + i];
                    var v: f32 = 0;
                    if (obs) {
                        v = (full[r * w + p * ps + i] - lo) / sc;
                        if (c.use_asinh and !(known and binary)) v = std.math.asinh(v);
                    }
                    dst[i] = v;
                    dst[ps + i] = if (obs) 0 else 1;
                }
            }
        }

        // Tokens: [Enc ; Dec]. Encoder tokens cover the whole window with covariates, else the
        // context. Only the decoder tokens of the horizon are materialized: the others read only
        // themselves and nothing reads them.
        const ne = n;
        const sl = ne + h;
        const x = try arena.alloc(f32, rows * sl * d);
        const enc = try arena.alloc(f32, rows * ne * d);
        try self.patch_proj.forward(io, arena, enc, emb_in, rows * ne);
        for (0..rows) |r| {
            @memcpy(x[r * sl * d ..][0 .. ne * d], enc[r * ne * d ..][0 .. ne * d]);
            for (0..h) |j| @memcpy(x[(r * sl + ne + j) * d ..][0..d], self.forecast_query);
        }

        // Key validity per (row, patch); first observed patch per series.
        const key_valid = try arena.alloc(bool, rows * ne);
        const first_obs = try arena.alloc(usize, in.groups);
        @memset(first_obs, ne);
        for (0..rows) |r| {
            const g = r / vall;
            var started = false;
            for (0..ne) |p| {
                var obs = false;
                for (0..ps) |i| obs = obs or full_obs[r * w + p * ps + i];
                if (obs) first_obs[g] = @min(first_obs[g], p);
                started = started or obs;
                key_valid[r * ne + p] = if (c.attend_missing_patches) (obs or started) else obs;
            }
        }
        for (first_obs) |*f| if (f.* == ne) {
            f.* = 0;
        };
        var keys_all_valid = true;
        for (key_valid) |v| keys_all_valid = keys_all_valid and v;

        const geo: Geo = .{
            .rows = rows,
            .groups = in.groups,
            .vall = vall,
            .ne = ne,
            .l = l,
            .h = h,
            .roles = in.roles,
            .cov = cov,
            .key_valid = key_valid,
            .keys_all_valid = keys_all_valid,
            .first_obs = first_obs,
        };
        try self.runTransformer(io, arena, x, geo);

        // Head on the target rows' decoder tokens.
        const n_rows_out = in.groups * n_tgt;
        const dec = try arena.alloc(f32, n_rows_out * h * d);
        var oi: usize = 0;
        for (0..rows) |r| {
            if (in.roles[r % vall] != .target) continue;
            nn.rmsNorm(io, dec[oi * h * d ..][0 .. h * d], x[(r * sl + ne) * d ..][0 .. h * d], h, d, self.out_norm, @floatCast(c.norm_eps));
            oi += 1;
        }
        const raw = try arena.alloc(f32, n_rows_out * hl * q);
        try self.head.forward(io, arena, raw, dec, n_rows_out * h);

        const m = c.medianIndex();
        const widths = try arena.alloc(f32, q);
        @memset(widths, 0);
        for (m + 1..q) |i| widths[i] = @floatCast(c.quantile_knots[i] - c.quantile_knots[i - 1]);
        for (0..m) |i| widths[i] = @floatCast(c.quantile_knots[i + 1] - c.quantile_knots[i]);
        const wf: f32 = @floatCast(c.quantile_width_factor);
        const big = std.math.floatMax(f32);

        oi = 0;
        for (0..rows) |r| {
            if (in.roles[r % vall] != .target) continue;
            // Anchor frame: the last context patch.
            const lo = loc[r * n + l - 1];
            const sc = scale[r * n + l - 1];
            for (0..hl) |sidx| {
                const rv = raw[(oi * hl + sidx) * q ..][0..q];
                const ov = out[(oi * hl + sidx) * q ..][0..q];
                // Monotone head: median free, the rest fan out by cumulative softplus increments.
                const med = rv[m];
                var acc: f32 = 0;
                for (m + 1..q) |i| {
                    acc += wf * widths[i] * nn.softplus(rv[i]);
                    ov[i] = std.math.clamp(med + acc, -big, big);
                }
                acc = 0;
                var i: usize = m;
                while (i > 0) {
                    i -= 1;
                    acc += wf * widths[i] * nn.softplus(rv[i]);
                    ov[i] = std.math.clamp(med - acc, -big, big);
                }
                ov[m] = med;
                // Back to data units: sinh(x) * scale + loc, then fix any overflow.
                for (ov) |*v| {
                    const y = if (c.use_asinh) std.math.sinh(v.*) else v.*;
                    v.* = y * sc + lo;
                }
                nn.clampNonfinite(ov);
            }
            oi += 1;
        }
    }

    fn runTransformer(self: *const Model, io: std.Io, arena: Allocator, x: []f32, geo: Geo) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const sl = geo.s();
        const rows = geo.rows;
        const eps: f32 = @floatCast(c.norm_eps);
        const hid: usize = c.mlp_hidden_dim;

        const xn = try arena.alloc(f32, rows * sl * d);
        const delta = try arena.alloc(f32, rows * sl * d);
        const mlp_h = try arena.alloc(f32, rows * sl * 2 * hid);
        const mlp_a = try arena.alloc(f32, rows * sl * hid);

        // RoPE tables per series: encoder token p sits at time p, decoder token j at time l + j,
        // both relative to the series' first observed patch.
        const hd: usize = c.headDim();
        const half = hd / 2;
        const cos_t = try arena.alloc(f32, geo.groups * sl * half);
        const sin_t = try arena.alloc(f32, geo.groups * sl * half);
        for (0..geo.groups) |g| {
            for (0..sl) |tok| {
                const time: i64 = if (tok < geo.ne) @intCast(tok) else @intCast(geo.l + tok - geo.ne);
                const pos: f32 = @floatFromInt(time - @as(i64, @intCast(geo.first_obs[g])));
                for (0..half) |i| {
                    const ang = self.inv_freq[i] * pos;
                    cos_t[(g * sl + tok) * half + i] = @cos(ang);
                    sin_t[(g * sl + tok) * half + i] = @sin(ang);
                }
            }
        }

        for (self.layers) |layer| {
            for (layer.attns[0..layer.n_attn]) |a| {
                nn.rmsNorm(io, xn, x, rows * sl, d, a.norm, eps);
                if (a.is_time) {
                    try self.timeAttention(io, arena, delta, xn, a, geo, cos_t, sin_t);
                } else {
                    try self.variateAttention(io, arena, delta, xn, a, geo);
                }
                nn.addInPlace(x, delta[0..x.len]);
            }
            nn.rmsNorm(io, xn, x, rows * sl, d, layer.mlp_norm, eps);
            nn.linear(io, mlp_h, xn, rows * sl, layer.w12, 2 * hid, d, null);
            const act: MlpAct = .{ .h = mlp_h, .a = mlp_a, .hid = hid };
            par.forRangeMin(io, rows * sl, 32, &act, MlpAct.run);
            nn.linear(io, delta, mlp_a, rows * sl, layer.w3, d, hid, null);
            nn.addInPlace(x, delta[0..x.len]);
        }
    }

    /// Time attention per row over `[Enc ; Dec]`. Encoder queries read valid encoder keys
    /// causally (diagonal always open); known-future rows read all valid keys. Decoder queries
    /// read the valid context keys and their own causal decoder chain.
    fn timeAttention(self: *const Model, io: std.Io, arena: Allocator, out: []f32, xn: []const f32, a: Attn, geo: Geo, cos_t: []const f32, sin_t: []const f32) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const rows = geo.rows;
        const sl = geo.s();

        // Q, K and V stay interleaved in one buffer: `(token, qkv, head, head_dim)`.
        const qkv = try arena.alloc(f32, rows * sl * 3 * d);
        nn.linear(io, qkv, xn, rows * sl, a.qkv, 3 * d, d, null);
        const oa = try arena.alloc(f32, rows * sl * d);

        const gate = if (a.gate) |gw| blk: {
            const g = try arena.alloc(f32, rows * sl * d);
            nn.linear(io, g, xn, rows * sl, gw, d, d, null);
            break :blk g;
        } else null;

        const job: TimeJob = .{
            .model = self,
            .geo = geo,
            .qkv = qkv,
            .o = oa,
            .cos_t = cos_t,
            .sin_t = sin_t,
        };
        // Phase 1: rotate q and k in place, per token.
        par.forRangeMin(io, rows * sl, 16, &job, TimeJob.rope);
        // Phase 2: attention per (row, head).
        par.forRange(io, rows * c.num_heads, &job, TimeJob.attend);
        // Phase 3: gate, then the output projection.
        if (gate) |gt| nn.mulSigmoid(oa, oa, gt);
        nn.linear(io, out, oa, rows * sl, a.out, d, d, null);
    }

    /// Variate attention: at every token slot, the rows of a series attend each other. A row
    /// reads another row on an encoder slot only if that row's patch is a valid key. With
    /// covariates two firewalls apply: a covariate row never reads a target row, and a
    /// known-future row never reads a causal row. The diagonal is always open.
    fn variateAttention(self: *const Model, io: std.Io, arena: Allocator, out: []f32, xn: []const f32, a: Attn, geo: Geo) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const rows = geo.rows;
        const sl = geo.s();

        const qkv = try arena.alloc(f32, rows * sl * 3 * d);
        nn.linear(io, qkv, xn, rows * sl, a.qkv, 3 * d, d, null);
        const gate = if (a.gate) |gw| blk: {
            const g = try arena.alloc(f32, rows * sl * d);
            nn.linear(io, g, xn, rows * sl, gw, d, d, null);
            break :blk g;
        } else null;

        const o = try arena.alloc(f32, rows * sl * d);
        const job: VariateJob = .{ .model = self, .geo = geo, .qkv = qkv, .o = o };
        par.forRangeMin(io, geo.groups * sl, 8, &job, VariateJob.run);
        if (gate) |gt| nn.mulSigmoid(o, o, gt);
        nn.linear(io, out[0 .. rows * sl * d], o, rows * sl, a.out, d, d, null);
    }
};

const TimeJob = struct {
    model: *const Model,
    geo: Geo,
    qkv: []f32,
    o: []f32,
    cos_t: []const f32,
    sin_t: []const f32,

    /// Tokens `[t0, t1)` of the packed `(row, token)` sequence: rotate q and k in place.
    /// The fused projection is stored `(token, qkv, head, head_dim)` (see `permuteQkv`), so a head
    /// is a plain contiguous slice. Partial RoPE: dims past rope_dim have zero frequency.
    fn rope(self: *const TimeJob, t0: usize, t1: usize) void {
        const c = self.model.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const half = hd / 2;
        const sl = self.geo.s();
        for (t0..t1) |idx| {
            const r = idx / sl;
            const tok = idx % sl;
            const g = r / self.geo.vall;
            const cs = self.cos_t[(g * sl + tok) * half ..][0..half];
            const sn = self.sin_t[(g * sl + tok) * half ..][0..half];
            const base = idx * 3 * d;
            for (0..nh) |hh| {
                nn.ropeHalf(self.qkv[base + hh * hd ..][0..hd], cs, sn, half);
                nn.ropeHalf(self.qkv[base + d + hh * hd ..][0..hd], cs, sn, half);
            }
        }
    }

    /// Items `[first, last)` of `(row, head)`.
    fn attend(self: *const TimeJob, first: usize, last: usize) void {
        const c = self.model.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const geo = self.geo;
        const ne = geo.ne;
        const l = geo.l;
        const h = geo.h;
        const sl = geo.s();
        const sm_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const ninf = -std.math.inf(f32);

        const scores = std.heap.smp_allocator.alloc(f32, @max(ne * ne, h * sl)) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(scores);

        for (first..last) |item| {
            const r = item / nh;
            const hh = item % nh;
            const off = hh * hd;
            const bidir = geo.cov and geo.role(r) == .known;
            const base = r * sl * 3 * d;
            const q = self.qkv[base + off ..];
            const k = self.qkv[base + d + off ..];
            const v = self.qkv[base + 2 * d + off ..];
            const o = self.o[r * sl * d + off ..];
            const kv = geo.key_valid[r * ne ..][0..ne];

            // Encoder queries over encoder keys.
            nn.gemmNT(scores, ne, q, 3 * d, k, 3 * d, ne, ne, hd, sm_scale);
            if (geo.keys_all_valid) {
                if (bidir) {
                    for (0..ne) |i| nn.softmax(scores[i * ne ..][0..ne]);
                } else {
                    // Causal: only a prefix of the row is scored; the tail stays zero.
                    for (0..ne) |i| {
                        nn.softmax(scores[i * ne ..][0 .. i + 1]);
                        @memset(scores[i * ne + i + 1 ..][0 .. ne - i - 1], 0);
                    }
                }
            } else {
                for (0..ne) |i| {
                    const row = scores[i * ne ..][0..ne];
                    for (0..ne) |j| {
                        const allowed = j == i or (kv[j] and (bidir or j <= i));
                        if (!allowed) row[j] = ninf;
                    }
                    nn.softmax(row);
                }
            }
            nn.gemmNN(o, d, scores, ne, v, 3 * d, ne, hd, ne);
            // Decoder queries: valid context keys, then the causal decoder chain.
            nn.gemmNT(scores, sl, self.qkv[base + ne * 3 * d + off ..], 3 * d, k, 3 * d, h, sl, hd, sm_scale);
            if (geo.keys_all_valid) {
                // The context keys are `[0, l)`; decoder queries also read their own causal chain,
                // which starts at `ne`. The gap `[l, ne)` (with covariates) stays out.
                for (0..h) |j| {
                    const row = scores[j * sl ..][0..sl];
                    if (l < ne) @memset(row[l..ne], ninf);
                    const end = ne + j + 1;
                    nn.softmax(row[0..end]);
                    @memset(row[end..sl], 0);
                }
            } else {
                for (0..h) |j| {
                    const row = scores[j * sl ..][0..sl];
                    for (0..ne) |kk| if (kk >= l or !kv[kk]) {
                        row[kk] = ninf;
                    };
                    for (ne + j + 1..sl) |kk| row[kk] = ninf;
                    nn.softmax(row);
                }
            }
            nn.gemmNN(o[ne * d ..], d, scores, sl, v, 3 * d, h, hd, sl);
        }
    }
};

const VariateJob = struct {
    model: *const Model,
    geo: Geo,
    qkv: []const f32,
    o: []f32,

    /// Items `[first, last)` of `(series, token slot)`. All rows of a series share one token, so
    /// the score matrix and the key mask are computed once per item and reused by every head.
    /// Heads are contiguous slices of the permuted projection, which makes both the dot products
    /// and the value accumulation plain vector loops.
    fn run(self: *const VariateJob, first: usize, last: usize) void {
        const c = self.model.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const geo = self.geo;
        const vall = geo.vall;
        const ne = geo.ne;
        const sl = geo.s();
        const sm_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const ninf = -std.math.inf(f32);
        const sc = std.heap.smp_allocator.alloc(f32, vall) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(sc);
        const allowed = std.heap.smp_allocator.alloc(bool, vall * vall) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(allowed);

        for (first..last) |item| {
            const g = item / sl;
            const tok = item % sl;
            const r0 = g * vall;
            for (0..vall) |iq| {
                for (0..vall) |ik| {
                    var ok = ik == iq;
                    if (!ok) {
                        const valid = tok >= ne or geo.key_valid[(r0 + ik) * ne + tok];
                        var forbidden = false;
                        if (geo.cov) {
                            const rq_role = geo.roles[iq];
                            const rk_role = geo.roles[ik];
                            forbidden = (rq_role != .target and rk_role == .target) or (rq_role == .known and rk_role != .known);
                        }
                        ok = valid and !forbidden;
                    }
                    allowed[iq * vall + ik] = ok;
                }
            }
            for (0..nh) |hh| {
                const qoff = hh * hd;
                for (0..vall) |iq| {
                    const qsrc = self.qkv[(r0 + iq) * sl * 3 * d + tok * 3 * d + qoff ..][0..hd];
                    const mask = allowed[iq * vall ..][0..vall];
                    for (0..vall) |ik| {
                        if (!mask[ik]) {
                            sc[ik] = ninf;
                            continue;
                        }
                        const ksrc = self.qkv[(r0 + ik) * sl * 3 * d + tok * 3 * d + d + qoff ..][0..hd];
                        sc[ik] = nn.dot(qsrc, ksrc) * sm_scale;
                    }
                    nn.softmax(sc);
                    const dst = self.o[(r0 + iq) * sl * d + tok * d + qoff ..][0..hd];
                    @memset(dst, 0);
                    for (0..vall) |ik| {
                        const w = sc[ik];
                        if (w == 0) continue;
                        const vsrc = self.qkv[(r0 + ik) * sl * 3 * d + tok * 3 * d + 2 * d + qoff ..][0..hd];
                        nn.mulAddInPlace(dst, vsrc, w);
                    }
                }
            }
        }
    }
};

const StatsJob = struct {
    model: *const Model,
    loc: []f32,
    scale: []f32,
    data: []const f32,
    observed: []const bool,
    t: usize,

    /// Rows `[r0, r1)`.
    fn run(self: *const StatsJob, r0: usize, r1: usize) void {
        const ps: usize = self.model.config.patch_size;
        const t = self.t;
        const n = t / ps;
        const min_scale = self.model.config.minimum_scale;
        for (r0..r1) |r| {
            var cum_w: f64 = 0;
            var cum_x: f64 = 0;
            var m2: f64 = 0;
            var prev_mean: f64 = 0;
            for (0..n) |p| {
                for (0..ps) |k| {
                    const i = p * ps + k;
                    const w: f64 = if (self.observed[r * t + i]) 1.0 else 0.0;
                    const x: f64 = self.data[r * t + i];
                    cum_w += w;
                    cum_x += w * x;
                    const den = @max(cum_w, 1.0);
                    const mean = cum_x / den;
                    m2 += (x - prev_mean) * (x - mean) * w;
                    prev_mean = mean;
                }
                const den = @max(cum_w, 1.0);
                const variance = m2 / @max(den - 1.0, 1.0);
                self.loc[r * n + p] = @floatCast(prev_mean);
                self.scale[r * n + p] = @floatCast(@max(@sqrt(variance), min_scale));
            }
        }
    }
};
