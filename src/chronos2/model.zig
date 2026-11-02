//! The Chronos-2 network (inference only): patch embedding, the time / group / feed-forward
//! encoder blocks and the quantile head. Mirrors `chronos.chronos2.model` and `.layers`.
//!
//! With the v1 univariate path every series is its own group, so group attention over the batch
//! axis has exactly one valid key per token and reduces to `x + o(v(norm(x)))`. The generic
//! additive-mask path is not needed.

const std = @import("std");
const cfg = @import("config.zig");
const core = @import("core");
const nn = core.nn;
const st = core.safetensors;

const Allocator = std.mem.Allocator;

const Linear = struct {
    w: []const f32,
    b: ?[]const f32 = null,
};

const ResBlock = struct {
    hidden: Linear,
    output: Linear,
    residual: Linear,
    in: usize,
    hid: usize,
    out: usize,

    fn load(io: std.Io, w: *st.Weights, buf: []u8, prefix: []const u8, in: usize, hid: usize, out: usize) !ResBlock {
        return .{
            .hidden = try loadLinear(io, w, buf, prefix, "hidden_layer", hid, in),
            .output = try loadLinear(io, w, buf, prefix, "output_layer", out, hid),
            .residual = try loadLinear(io, w, buf, prefix, "residual_layer", out, in),
            .in = in,
            .hid = hid,
            .out = out,
        };
    }

    /// `output(relu(hidden(x))) + residual(x)` for `rows` input vectors.
    fn forward(self: ResBlock, io: std.Io, allocator: Allocator, y: []f32, x: []const f32, rows: usize) !void {
        const h = try allocator.alloc(f32, rows * self.hid);
        defer allocator.free(h);
        nn.linear(io, h, x, rows, self.hidden.w, self.hid, self.in, self.hidden.b);
        nn.relu(h);
        nn.linear(io, y, h, rows, self.output.w, self.out, self.hid, self.output.b);
        const r = try allocator.alloc(f32, rows * self.out);
        defer allocator.free(r);
        nn.linear(io, r, x, rows, self.residual.w, self.out, self.in, self.residual.b);
        for (y[0 .. rows * self.out], r) |*a, b| a.* += b;
    }
};

const Attn = struct {
    norm: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []const f32,
};

const Layer = struct {
    time: Attn,
    group: Attn,
    mlp_norm: []const f32,
    wi: []const f32,
    wo: []const f32,
};

fn tensorName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

fn loadLinear(io: std.Io, w: *st.Weights, buf: []u8, prefix: []const u8, name: []const u8, out: usize, in: usize) !Linear {
    const wn = tensorName(buf, "{s}.{s}.weight", .{ prefix, name });
    const weight = try w.get(io, wn, out * in);
    const bn = tensorName(buf, "{s}.{s}.bias", .{ prefix, name });
    const bias = try w.get(io, bn, out);
    return .{ .w = weight, .b = bias };
}

fn loadAttn(io: std.Io, w: *st.Weights, buf: []u8, layer: usize, which: usize, d: usize) !Attn {
    // A private prefix buffer: `buf` is reused for the tensor names below.
    var pbuf: [128]u8 = undefined;
    const p = tensorName(&pbuf, "encoder.block.{d}.layer.{d}.", .{ layer, which });
    return .{
        .norm = try w.get(io, tensorName(buf, "{s}layer_norm.weight", .{p}), d),
        .q = try w.get(io, tensorName(buf, "{s}self_attention.q.weight", .{p}), d * d),
        .k = try w.get(io, tensorName(buf, "{s}self_attention.k.weight", .{p}), d * d),
        .v = try w.get(io, tensorName(buf, "{s}self_attention.v.weight", .{p}), d * d),
        .o = try w.get(io, tensorName(buf, "{s}self_attention.o.weight", .{p}), d * d),
    };
}

pub const Model = struct {
    config: cfg.Config,
    input_embed: ResBlock,
    layers: []Layer,
    final_norm: []const f32,
    output_embed: ResBlock,
    reg: []const f32,
    inv_freq: []f32,

    pub fn load(io: std.Io, allocator: Allocator, config: cfg.Config, w: *st.Weights) !Model {
        const d: usize = config.d_model;
        const ps: usize = config.chronos_config.input_patch_size;
        const hid: usize = config.d_ff;
        const q: usize = config.numQuantiles();
        var buf: [256]u8 = undefined;

        const input_embed = try ResBlock.load(io, w, &buf, "input_patch_embedding", 3 * ps, hid, d);
        const output_embed = try ResBlock.load(io, w, &buf, "output_patch_embedding", d, hid, q * ps);
        const final_norm = try w.get(io, "encoder.final_layer_norm.weight", d);
        const shared = try w.get(io, "shared.weight", 2 * d);
        const reg = shared[d .. 2 * d];

        const layers = try allocator.alloc(Layer, config.num_layers);
        errdefer allocator.free(layers);
        for (0..config.num_layers) |li| {
            const time = try loadAttn(io, w, &buf, li, 0, d);
            const group = try loadAttn(io, w, &buf, li, 1, d);
            const mlp_norm = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.2.layer_norm.weight", .{li}), d);
            const wi = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.2.mlp.wi.weight", .{li}), hid * d);
            const wo = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.2.mlp.wo.weight", .{li}), d * hid);
            layers[li] = .{ .time = time, .group = group, .mlp_norm = mlp_norm, .wi = wi, .wo = wo };
        }

        const hd: usize = config.headDim();
        const inv_freq = try allocator.alloc(f32, hd / 2);
        const base: f32 = @floatCast(config.rope_theta);
        for (0..hd / 2) |i| {
            const idx: f32 = @floatFromInt(2 * i);
            inv_freq[i] = 1.0 / std.math.pow(f32, base, idx / @as(f32, @floatFromInt(hd)));
        }

        return .{
            .config = config,
            .input_embed = input_embed,
            .layers = layers,
            .final_norm = final_norm,
            .output_embed = output_embed,
            .reg = reg,
            .inv_freq = inv_freq,
        };
    }

    pub fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.layers);
        allocator.free(self.inv_freq);
    }

    /// Embed `(rows, 3 * patch_size)` patch features with the input residual block.
    pub fn embed(self: *const Model, io: std.Io, allocator: Allocator, out: []f32, feats: []const f32, rows: usize) !void {
        try self.input_embed.forward(io, allocator, out, feats, rows);
    }

    /// Run the encoder over `x` `(b, seq, d)` in place. `key_valid` is `(b, seq)`: a key is
    /// attended only where it is set (context patches with an observation, REG and future).
    pub fn encode(
        self: *const Model,
        io: std.Io,
        allocator: Allocator,
        x: []f32,
        key_valid: []const bool,
        b: usize,
        n_ctx: usize,
        n_out: usize,
    ) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const hid: usize = c.d_ff;
        const hd: usize = c.headDim();
        const half = hd / 2;
        const seq = n_ctx + 1 + n_out;
        const rows = b * seq;
        const eps: f32 = @floatCast(c.layer_norm_epsilon);

        const xn = try allocator.alloc(f32, rows * d);
        const delta = try allocator.alloc(f32, rows * d);
        const mlp_h = try allocator.alloc(f32, rows * hid);
        const qb = try allocator.alloc(f32, rows * d);
        const kb = try allocator.alloc(f32, rows * d);
        const vb = try allocator.alloc(f32, rows * d);
        const ab = try allocator.alloc(f32, rows * d);

        const cos_t = try allocator.alloc(f32, seq * half);
        const sin_t = try allocator.alloc(f32, seq * half);
        for (0..seq) |tok| {
            const pos: f32 = @floatFromInt(tok);
            for (0..half) |i| {
                const ang = self.inv_freq[i] * pos;
                cos_t[tok * half + i] = @cos(ang);
                sin_t[tok * half + i] = @sin(ang);
            }
        }

        for (self.layers) |layer| {
            nn.rmsNorm(io, xn, x, rows, d, layer.time.norm, eps);
            try self.timeAttention(io, delta, xn, layer.time, qb, kb, vb, ab, cos_t, sin_t, key_valid, b, seq);
            for (x, delta[0..x.len]) |*xv, dv| xv.* += dv;

            nn.rmsNorm(io, xn, x, rows, d, layer.group.norm, eps);
            self.groupAttention(io, delta, xn, layer.group, vb, rows);
            for (x, delta[0..x.len]) |*xv, dv| xv.* += dv;

            nn.rmsNorm(io, xn, x, rows, d, layer.mlp_norm, eps);
            nn.linear(io, mlp_h, xn, rows, layer.wi, hid, d, null);
            nn.relu(mlp_h);
            nn.linear(io, delta, mlp_h, rows, layer.wo, d, hid, null);
            for (x, delta[0..x.len]) |*xv, dv| xv.* += dv;
        }

        nn.rmsNorm(io, x, x, rows, d, self.final_norm, eps);
    }

    fn timeAttention(
        self: *const Model,
        io: std.Io,
        out: []f32,
        xn: []const f32,
        a: Attn,
        qb: []f32,
        kb: []f32,
        vb: []f32,
        ab: []f32,
        cos_t: []const f32,
        sin_t: []const f32,
        key_valid: []const bool,
        b: usize,
        seq: usize,
    ) !void {
        const c = self.config;
        const d: usize = c.d_model;
        const nh: usize = c.num_heads;
        const hd: usize = c.headDim();
        const half = hd / 2;
        const rows = b * seq;

        nn.linear(io, qb, xn, rows, a.q, d, d, null);
        nn.linear(io, kb, xn, rows, a.k, d, d, null);
        nn.linear(io, vb, xn, rows, a.v, d, d, null);

        for (0..rows) |r| {
            const tok = r % seq;
            const cs = cos_t[tok * half ..][0..half];
            const sn = sin_t[tok * half ..][0..half];
            for (0..nh) |hh| {
                const off = r * d + hh * hd;
                nn.ropeHalf(qb[off..][0..hd], cs, sn, half);
                nn.ropeHalf(kb[off..][0..hd], cs, sn, half);
            }
        }

        const job: TimeJob = .{
            .q = qb,
            .k = kb,
            .v = vb,
            .o = ab,
            .key_valid = key_valid,
            .seq = seq,
            .d = d,
            .nh = nh,
            .hd = hd,
        };
        core.par.forRange(io, b * nh, &job, TimeJob.attend);
        nn.linear(io, out, ab, rows, a.o, d, d, null);
    }

    /// Group attention with singleton groups: `x + o(v(norm(x)))` per token.
    fn groupAttention(self: *const Model, io: std.Io, out: []f32, xn: []const f32, a: Attn, vb: []f32, rows: usize) void {
        const d: usize = self.config.d_model;
        nn.linear(io, vb, xn, rows, a.v, d, d, null);
        nn.linear(io, out, vb, rows, a.o, d, d, null);
    }

    /// Quantile head on the last `n_out` tokens of each series: `(rows, num_quantiles * patch_size)`.
    pub fn outputHead(self: *const Model, io: std.Io, allocator: Allocator, out: []f32, hidden: []const f32, rows: usize) !void {
        try self.output_embed.forward(io, allocator, out, hidden, rows);
    }
};

const TimeJob = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []f32,
    key_valid: []const bool,
    seq: usize,
    d: usize,
    nh: usize,
    hd: usize,

    /// Items `[first, last)` of `(series, head)`.
    fn attend(self: *const TimeJob, first: usize, last: usize) void {
        const seq = self.seq;
        const d = self.d;
        const hd = self.hd;
        const mask_min: f32 = -std.math.floatMax(f32);
        const scores = std.heap.smp_allocator.alloc(f32, seq * seq) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(scores);

        for (first..last) |item| {
            const r = item / self.nh;
            const hh = item % self.nh;
            const off = hh * hd;
            const base = r * seq * d;
            const kv = self.key_valid[r * seq ..][0..seq];

            nn.gemmNT(scores, seq, self.q[base + off ..], d, self.k[base + off ..], d, seq, seq, hd, 1.0);
            for (0..seq) |i| {
                const row = scores[i * seq ..][0..seq];
                for (0..seq) |j| if (!kv[j]) {
                    row[j] += mask_min;
                };
                nn.softmax(row);
            }
            nn.gemmNN(self.o[base + off ..], d, scores, seq, self.v[base + off ..], d, seq, hd, seq);
        }
    }
};
