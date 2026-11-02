//! Chronos-2 inference configuration, parsed from the checkpoint's `config.json`.

const std = @import("std");

pub const Config = struct {
    d_model: u32,
    d_ff: u32,
    num_heads: u32,
    d_kv: u32,
    num_layers: u32,
    layer_norm_epsilon: f64 = 1e-6,
    rope_theta: f64 = 10000.0,
    dense_act_fn: []const u8 = "relu",
    chronos_config: Chronos,

    pub const Chronos = struct {
        context_length: u32,
        input_patch_size: u32,
        input_patch_stride: u32,
        output_patch_size: u32,
        max_output_patches: u32,
        quantiles: []const f64,
        time_encoding_scale: f64,
        use_arcsinh: bool = true,
        use_reg_token: bool = true,
    };

    pub fn headDim(self: Config) usize {
        return self.d_kv;
    }

    pub fn numQuantiles(self: Config) usize {
        return self.chronos_config.quantiles.len;
    }

    pub fn predictionLength(self: Config) usize {
        return @as(usize, self.chronos_config.max_output_patches) * self.chronos_config.output_patch_size;
    }

    pub fn validate(self: Config) !void {
        const ch = self.chronos_config;
        if (self.num_heads == 0 or self.d_model != self.num_heads * self.d_kv) return error.InvalidConfig;
        if (self.d_ff == 0 or self.num_layers == 0) return error.InvalidConfig;
        if (ch.input_patch_size == 0 or ch.input_patch_size != ch.output_patch_size) return error.InvalidConfig;
        if (ch.input_patch_stride != ch.input_patch_size) return error.InvalidConfig;
        if (ch.context_length == 0 or ch.max_output_patches == 0) return error.InvalidConfig;
        if (ch.quantiles.len == 0) return error.InvalidConfig;
        if (!std.mem.eql(u8, self.dense_act_fn, "relu")) return error.UnsupportedActivation;
        if (!ch.use_reg_token) return error.UnsupportedConfig;
        if (ch.time_encoding_scale <= 0 or self.layer_norm_epsilon <= 0) return error.InvalidConfig;
    }
};

pub const Parsed = std.json.Parsed(Config);

pub fn parse(allocator: std.mem.Allocator, text: []const u8) !Parsed {
    const parsed = try std.json.parseFromSlice(Config, allocator, text, .{ .ignore_unknown_fields = true });
    errdefer parsed.deinit();
    try parsed.value.validate();
    return parsed;
}

/// Whether `config.json` describes a Chronos-2 checkpoint (`architectures` contains
/// `Chronos2Model`).
pub fn isChronos2(allocator: std.mem.Allocator, text: []const u8) !bool {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    const arch = parsed.value.object.get("architectures") orelse return false;
    for (arch.array.items) |item| {
        if (item == .string and std.mem.eql(u8, item.string, "Chronos2Model")) return true;
    }
    return false;
}
