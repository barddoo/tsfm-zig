//! The Chronos-Bolt network (inference only): a T5 encoder over patch embeddings, a one-token T5
//! decoder that cross-attends to it, and a direct multi-quantile head. Mirrors
//! `chronos.chronos_bolt.ChronosBoltModelForForecasting` and the `transformers` T5 stack it wraps.
//!
//! T5 specifics: RMS-style layer norm without bias, attention scores are *not* scaled by
//! `1/sqrt(head_dim)`, positions enter only through a learned relative-position bias that is
//! computed once (block 0) and shared by every encoder layer, and the feed-forward is ReLU.
//!
//! The decoder sees one token, so its self-attention softmax is over a single key and equals
//! `o(v(norm(x)))`; its cross-attention has no position bias, only the encoder key mask.

const std = @import("std");
const cfg = @import("config.zig");
const core = @import("core");
const nn = core.nn;
const st = core.safetensors;
const ResBlock = core.resblock.ResBlock;

const Allocator = std.mem.Allocator;

const EncLayer = struct {
    attn_norm: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []const f32,
    ffn_norm: []const f32,
    wi: []const f32,
    wo: []const f32,
};

const DecLayer = struct {
    self_norm: []const f32,
    self_v: []const f32,
    self_o: []const f32,
    cross_norm: []const f32,
    cross_q: []const f32,
    cross_k: []const f32,
    cross_v: []const f32,
    cross_o: []const f32,
    ffn_norm: []const f32,
    wi: []const f32,
    wo: []const f32,
};

fn tensorName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// T5's bidirectional relative-position bucket for `rel = key_pos - query_pos`
/// (`T5Attention._relative_position_bucket`, including its float32 log for the large buckets).
pub fn relativeBucket(rel: i64, num_buckets: u32, max_distance: u32) usize {
    const half: u32 = num_buckets / 2;
    const base: u32 = if (rel > 0) half else 0;
    const rp: u32 = @intCast(@abs(rel));
    const max_exact: u32 = half / 2;
    if (rp < max_exact) return base + rp;
    const ratio: f32 = @as(f32, @floatFromInt(rp)) / @as(f32, @floatFromInt(max_exact));
    const denom: f32 = @floatCast(@log(@as(f64, @floatFromInt(max_distance)) / @as(f64, @floatFromInt(max_exact))));
    const large_f: f32 = @log(ratio) / denom * @as(f32, @floatFromInt(half - max_exact));
    const large: u32 = max_exact + @as(u32, @intFromFloat(large_f));
    return base + @min(large, half - 1);
}

pub const Model = struct {
    config: cfg.Config,
    input_embed: ResBlock,
    output_embed: ResBlock,
    /// `(vocab, d_model)`: row `decoder_start_token_id` starts the decoder, row `reg_token_id` is [REG].
    shared: []const f32,
    enc_layers: []EncLayer,
    enc_final_norm: []const f32,
    /// `(num_buckets, num_heads)` relative-position bias table of encoder block 0.
    rel_bias: []const f32,
    dec_layers: []DecLayer,
    dec_final_norm: []const f32,

    pub fn load(io: std.Io, allocator: Allocator, config: cfg.Config, w: *st.Weights) !Model {
        const d: usize = config.d_model;
        const ff: usize = config.d_ff;
        const inner = config.inner();
        const ps: usize = config.chronos_config.input_patch_size;
        const nq = config.numQuantiles();
        const out_dim = nq * @as(usize, config.chronos_config.prediction_length);
        var buf: [256]u8 = undefined;

        const input_embed = try ResBlock.load(io, w, &buf, "input_patch_embedding", 2 * ps, ff, d);
        const output_embed = try ResBlock.load(io, w, &buf, "output_patch_embedding", d, ff, out_dim);
        const shared = try w.get(io, "shared.weight", config.vocabSize() * d);
        const rel_bias = try w.get(io, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", @as(usize, config.relative_attention_num_buckets) * config.num_heads);
        const enc_final = try w.get(io, "encoder.final_layer_norm.weight", d);
        const dec_final = try w.get(io, "decoder.final_layer_norm.weight", d);

        const enc = try allocator.alloc(EncLayer, config.num_layers);
        errdefer allocator.free(enc);
        for (enc, 0..) |*l, i| {
            l.* = .{
                .attn_norm = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.0.layer_norm.weight", .{i}), d),
                .q = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.0.SelfAttention.q.weight", .{i}), inner * d),
                .k = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.0.SelfAttention.k.weight", .{i}), inner * d),
                .v = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.0.SelfAttention.v.weight", .{i}), inner * d),
                .o = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.0.SelfAttention.o.weight", .{i}), d * inner),
                .ffn_norm = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.1.layer_norm.weight", .{i}), d),
                .wi = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.1.DenseReluDense.wi.weight", .{i}), ff * d),
                .wo = try w.get(io, tensorName(&buf, "encoder.block.{d}.layer.1.DenseReluDense.wo.weight", .{i}), d * ff),
            };
        }

        const dec = try allocator.alloc(DecLayer, config.num_decoder_layers);
        errdefer allocator.free(dec);
        for (dec, 0..) |*l, i| {
            l.* = .{
                .self_norm = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.0.layer_norm.weight", .{i}), d),
                .self_v = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.0.SelfAttention.v.weight", .{i}), inner * d),
                .self_o = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.0.SelfAttention.o.weight", .{i}), d * inner),
                .cross_norm = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.1.layer_norm.weight", .{i}), d),
                .cross_q = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.1.EncDecAttention.q.weight", .{i}), inner * d),
                .cross_k = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.1.EncDecAttention.k.weight", .{i}), inner * d),
                .cross_v = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.1.EncDecAttention.v.weight", .{i}), inner * d),
                .cross_o = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.1.EncDecAttention.o.weight", .{i}), d * inner),
                .ffn_norm = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.2.layer_norm.weight", .{i}), d),
                .wi = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.2.DenseReluDense.wi.weight", .{i}), ff * d),
                .wo = try w.get(io, tensorName(&buf, "decoder.block.{d}.layer.2.DenseReluDense.wo.weight", .{i}), d * ff),
            };
        }

        return .{
            .config = config,
            .input_embed = input_embed,
            .output_embed = output_embed,
            .shared = shared,
            .enc_layers = enc,
            .enc_final_norm = enc_final,
            .rel_bias = rel_bias,
            .dec_layers = dec,
            .dec_final_norm = dec_final,
        };
    }

    pub fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.enc_layers);
        allocator.free(self.dec_layers);
    }

    /// Embed `(rows, 2 * patch_size)` patch features (values, then observation mask).
    pub fn embed(self: *const Model, io: std.Io, allocator: Allocator, out: []f32, feats: []const f32, rows: usize) !void {
        try self.input_embed.forward(io, allocator, out, feats, rows);
    }

    /// The [REG] token embedding.
    pub fn regEmbedding(self: *const Model) []const f32 {
        const d: usize = self.config.d_model;
        return self.shared[@as(usize, self.config.reg_token_id) * d ..][0..d];
    }

    /// `(num_heads, seq, seq)` additive bias: `bias[h][i][j] = table[bucket(j - i)][h]`.
    fn positionBias(self: *const Model, out: []f32, seq: usize) void {
        const c = self.config;
        const nh: usize = c.num_heads;
        for (0..seq) |i| {
            for (0..seq) |j| {
                const rel: i64 = @as(i64, @intCast(j)) - @as(i64, @intCast(i));
                const bucket = relativeBucket(rel, c.relative_attention_num_buckets, c.relative_attention_max_distance);
                for (0..nh) |h| out[h * seq * seq + i * seq + j] = self.rel_bias[bucket * nh + h];
            }
        }
    }

    /// Run the encoder over `x` `(nb, seq, d)` in place. `key_valid` is `(nb, seq)`: a key is
    /// attended only where it is set (patches with an observation, and [REG]). All scratch lives
    /// in a per-call arena over `allocator`.
    pub fn encode(self: *const Model, io: std.Io, allocator: Allocator, x: []f32, key_valid: []const bool, nb: usize, seq: usize) !void {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const c = self.config;
        const d: usize = c.d_model;
        const ff: usize = c.d_ff;
        const inner = c.inner();
        const nh: usize = c.num_heads;
        const hd: usize = c.d_kv;
        const rows = nb * seq;
        const eps: f32 = @floatCast(c.layer_norm_epsilon);

        const xn = try arena.alloc(f32, rows * d);
        const delta = try arena.alloc(f32, rows * d);
        const mlp_h = try arena.alloc(f32, rows * ff);
        const qb = try arena.alloc(f32, rows * inner);
        const kb = try arena.alloc(f32, rows * inner);
        const vb = try arena.alloc(f32, rows * inner);
        const ab = try arena.alloc(f32, rows * inner);
        const bias = try arena.alloc(f32, nh * seq * seq);
        self.positionBias(bias, seq);

        var keys_all_valid = true;
        for (key_valid) |v| keys_all_valid = keys_all_valid and v;

        for (self.enc_layers) |l| {
            nn.rmsNorm(io, xn, x, rows, d, l.attn_norm, eps);
            nn.linear(io, qb, xn, rows, l.q, inner, d, null);
            nn.linear(io, kb, xn, rows, l.k, inner, d, null);
            nn.linear(io, vb, xn, rows, l.v, inner, d, null);
            const job: SelfJob = .{ .q = qb, .k = kb, .v = vb, .o = ab, .bias = bias, .key_valid = key_valid, .keys_all_valid = keys_all_valid, .seq = seq, .inner = inner, .nh = nh, .hd = hd };
            core.par.forRange(io, nb * nh, &job, SelfJob.attend);
            nn.linear(io, delta, ab, rows, l.o, d, inner, null);
            nn.addInPlace(x, delta[0..x.len]);

            nn.rmsNorm(io, xn, x, rows, d, l.ffn_norm, eps);
            nn.linear(io, mlp_h, xn, rows, l.wi, ff, d, null);
            nn.relu(mlp_h);
            nn.linear(io, delta, mlp_h, rows, l.wo, d, ff, null);
            nn.addInPlace(x, delta[0..x.len]);
        }
        nn.rmsNorm(io, x, x, rows, d, self.enc_final_norm, eps);
    }

    /// Run the one-token decoder for `nb` series against the encoder output `enc` `(nb, seq, d)`;
    /// writes the final hidden state `(nb, d)` to `out`.
    pub fn decode(self: *const Model, io: std.Io, allocator: Allocator, out: []f32, enc: []const f32, key_valid: []const bool, nb: usize, seq: usize) !void {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const c = self.config;
        const d: usize = c.d_model;
        const ff: usize = c.d_ff;
        const inner = c.inner();
        const nh: usize = c.num_heads;
        const hd: usize = c.d_kv;
        const eps: f32 = @floatCast(c.layer_norm_epsilon);

        const x = try arena.alloc(f32, nb * d);
        const xn = try arena.alloc(f32, nb * d);
        const delta = try arena.alloc(f32, nb * d);
        const mlp_h = try arena.alloc(f32, nb * ff);
        const qd = try arena.alloc(f32, nb * inner);
        const vd = try arena.alloc(f32, nb * inner);
        const ad = try arena.alloc(f32, nb * inner);
        const kb = try arena.alloc(f32, nb * seq * inner);
        const vb = try arena.alloc(f32, nb * seq * inner);

        const start = self.shared[@as(usize, c.decoder_start_token_id) * d ..][0..d];
        for (0..nb) |r| @memcpy(x[r * d ..][0..d], start);

        for (self.dec_layers) |l| {
            // Self-attention over the single decoder token: softmax over one key is 1.
            nn.rmsNorm(io, xn, x, nb, d, l.self_norm, eps);
            nn.linear(io, vd, xn, nb, l.self_v, inner, d, null);
            nn.linear(io, delta, vd, nb, l.self_o, d, inner, null);
            nn.addInPlace(x, delta);

            // Cross-attention to the encoder output; no position bias, only the key mask.
            nn.rmsNorm(io, xn, x, nb, d, l.cross_norm, eps);
            nn.linear(io, qd, xn, nb, l.cross_q, inner, d, null);
            nn.linear(io, kb, enc, nb * seq, l.cross_k, inner, d, null);
            nn.linear(io, vb, enc, nb * seq, l.cross_v, inner, d, null);
            const job: CrossJob = .{ .q = qd, .k = kb, .v = vb, .o = ad, .key_valid = key_valid, .seq = seq, .inner = inner, .nh = nh, .hd = hd };
            core.par.forRange(io, nb * nh, &job, CrossJob.attend);
            nn.linear(io, delta, ad, nb, l.cross_o, d, inner, null);
            nn.addInPlace(x, delta);

            nn.rmsNorm(io, xn, x, nb, d, l.ffn_norm, eps);
            nn.linear(io, mlp_h, xn, nb, l.wi, ff, d, null);
            nn.relu(mlp_h);
            nn.linear(io, delta, mlp_h, nb, l.wo, d, ff, null);
            nn.addInPlace(x, delta);
        }
        nn.rmsNorm(io, out, x, nb, d, self.dec_final_norm, eps);
    }

    /// Quantile head: `(rows, d)` decoder states to `(rows, num_quantiles * prediction_length)`.
    pub fn outputHead(self: *const Model, io: std.Io, allocator: Allocator, out: []f32, hidden: []const f32, rows: usize) !void {
        try self.output_embed.forward(io, allocator, out, hidden, rows);
    }
};

/// Encoder self-attention, one `(series, head)` per item, through the shared comptime-masked kernel.
const SelfJob = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []f32,
    bias: []const f32, // (nh, seq, seq)
    key_valid: []const bool,
    keys_all_valid: bool,
    seq: usize,
    inner: usize,
    nh: usize,
    hd: usize,

    fn attend(self: *const SelfJob, first: usize, last: usize) void {
        const seq = self.seq;
        const scores = std.heap.smp_allocator.alloc(f32, seq * seq) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(scores);
        // T5 masks invalid keys with an additive -floatMax and does not scale the scores.
        const mask = core.attn.Mask.pick(false, self.keys_all_valid, true);
        const hd = self.hd;
        const strides: core.attn.Strides = .{ .q = self.inner, .k = self.inner, .v = self.inner, .o = self.inner };
        for (first..last) |item| {
            const r = item / self.nh;
            const h = item % self.nh;
            const off = r * seq * self.inner + h * hd;
            core.attn.selfAttendRuntime(mask, scores, self.q[off..], self.k[off..], self.v[off..], self.o[off..], strides, seq, self.hd, 1.0, self.key_valid[r * seq ..][0..seq], self.bias[h * seq * seq ..][0 .. seq * seq]);
        }
    }
};

/// Decoder cross-attention for one query token, one `(series, head)` per item.
const CrossJob = struct {
    q: []const f32, // (nb, inner)
    k: []const f32, // (nb, seq, inner)
    v: []const f32,
    o: []f32, // (nb, inner)
    key_valid: []const bool,
    seq: usize,
    inner: usize,
    nh: usize,
    hd: usize,

    fn attend(self: *const CrossJob, first: usize, last: usize) void {
        const seq = self.seq;
        const hd = self.hd;
        const mask_min: f32 = -std.math.floatMax(f32);
        const scores = std.heap.smp_allocator.alloc(f32, seq) catch @panic("out of memory");
        defer std.heap.smp_allocator.free(scores);
        for (first..last) |item| {
            const r = item / self.nh;
            const h = item % self.nh;
            const qh = self.q[r * self.inner + h * hd ..][0..hd];
            for (0..seq) |j| {
                const kh = self.k[(r * seq + j) * self.inner + h * hd ..][0..hd];
                scores[j] = nn.dot(qh, kh);
                if (!self.key_valid[r * seq + j]) scores[j] += mask_min;
            }
            nn.softmax(scores);
            const oh = self.o[r * self.inner + h * hd ..][0..hd];
            @memset(oh, 0);
            for (0..seq) |j| nn.mulAddInPlace(oh, self.v[(r * seq + j) * self.inner + h * hd ..][0..hd], scores[j]);
        }
    }
};

// Bucket ids from `T5Attention._relative_position_bucket(arange(-140, 141), bidirectional=True,
// num_buckets=32, max_distance=128)` in transformers, so the float32 log boundaries are checked.
test "relativeBucket matches transformers for -140..140" {
    const expected = [_]u8{ 15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,12,12,12,12,12,12,12,12,12,12,12,12,12,12,11,11,11,11,11,11,11,11,11,10,10,10,10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31 };
    for (expected, 0..) |want, i| {
        const rel: i64 = @as(i64, @intCast(i)) - 140;
        try std.testing.expectEqual(@as(usize, want), relativeBucket(rel, 32, 128));
    }
}
