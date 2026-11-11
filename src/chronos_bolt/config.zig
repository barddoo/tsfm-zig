//! Chronos-Bolt inference configuration, parsed from the checkpoint's `config.json` (a T5 config
//! plus a `chronos_config` block).

const std = @import("std");
const core = @import("core");

pub const architecture = "ChronosBoltModelForForecasting";

pub const Config = struct {
    d_model: u32,
    d_ff: u32,
    num_heads: u32,
    d_kv: u32,
    num_layers: u32,
    num_decoder_layers: u32,
    layer_norm_epsilon: f64 = 1e-6,
    dense_act_fn: []const u8 = "relu",
    relative_attention_num_buckets: u32 = 32,
    relative_attention_max_distance: u32 = 128,
    decoder_start_token_id: u32 = 0,
    reg_token_id: u32 = 1,
    chronos_config: Chronos,

    pub const Chronos = struct {
        context_length: u32,
        input_patch_size: u32,
        input_patch_stride: u32,
        prediction_length: u32,
        quantiles: []const f64,
        use_reg_token: bool = false,
    };

    /// Width of the attention projections (`num_heads * d_kv`; equals `d_model` in the released models).
    pub fn inner(self: Config) usize {
        return @as(usize, self.num_heads) * self.d_kv;
    }

    pub fn numQuantiles(self: Config) usize {
        return self.chronos_config.quantiles.len;
    }

    /// Rows of `shared.weight`: the decoder start token and, with `use_reg_token`, the [REG] token.
    pub fn vocabSize(self: Config) usize {
        return if (self.chronos_config.use_reg_token) 2 else 1;
    }

    pub fn validate(self: Config) !void {
        const c = self.chronos_config;
        if (self.d_model == 0 or self.d_ff == 0 or self.num_heads == 0 or self.d_kv == 0) return error.InvalidConfig;
        if (self.num_layers == 0 or self.num_decoder_layers == 0) return error.InvalidConfig;
        if (c.input_patch_size == 0 or c.prediction_length == 0 or c.context_length == 0) return error.InvalidConfig;
        if (c.quantiles.len == 0) return error.InvalidConfig;
        // Patches are non-overlapping in every released checkpoint; overlapping ones are not ported.
        if (c.input_patch_stride != c.input_patch_size) return error.UnsupportedPatchStride;
        if (!std.mem.eql(u8, self.dense_act_fn, "relu")) return error.UnsupportedActivation;
        if (self.relative_attention_num_buckets < 4 or self.relative_attention_num_buckets % 2 != 0) return error.InvalidConfig;
        if (self.reg_token_id >= self.vocabSize() or self.decoder_start_token_id >= self.vocabSize()) return error.InvalidConfig;
    }
};

pub const Parsed = std.json.Parsed(Config);

pub fn parse(allocator: std.mem.Allocator, text: []const u8) !Parsed {
    const parsed = try std.json.parseFromSlice(Config, allocator, text, .{ .ignore_unknown_fields = true });
    errdefer parsed.deinit();
    try parsed.value.validate();
    return parsed;
}

pub fn isChronosBolt(allocator: std.mem.Allocator, text: []const u8) !bool {
    return core.family.hasArchitecture(allocator, text, architecture);
}
