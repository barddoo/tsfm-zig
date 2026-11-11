//! Chronos-2 as a registered model family (v1: univariate, no covariates).

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const core = @import("core");
const config = @import("config.zig");
const model_mod = @import("model.zig");
const forecaster = @import("forecaster.zig");

const family = core.family;
const series_io = core.series;
const st = core.safetensors;

pub const name = "Chronos-2";

pub fn matches(allocator: mem.Allocator, config_text: []const u8) !bool {
    return config.isChronos2(allocator, config_text);
}

pub fn run(req: *const family.Request) !family.Forecast {
    const allocator = req.allocator;
    const io = req.io;

    if (req.covariates.any()) {
        Io.File.stderr().writeStreamingAll(io, "Chronos-2 does not support covariates in this runner (v1): --past, --known, --cat-past and --cat-known are unavailable\n") catch {};
        return error.CovariatesUnsupported;
    }
    const o = req.options;
    if (o.max_context != null or o.block_patches != null or o.no_symmetric or o.no_clip) {
        Io.File.stderr().writeStreamingAll(io, "--max-context, --block-patches, --no-symmetric and --no-clip are Timestral-only options\n") catch {};
        return error.UnsupportedOption;
    }

    const parsed = try config.parse(allocator, req.config_text);
    defer parsed.deinit();
    const c = parsed.value;
    if (req.horizon > c.predictionLength()) {
        std.debug.print("Chronos-2 supports horizons up to {d} steps (long-horizon unrolling is not ported)\n", .{c.predictionLength()});
        return error.HorizonTooLong;
    }

    const w_path = try allocator.print("{s}/model.safetensors", .{req.model_dir});
    defer allocator.free(w_path);
    var weights = try st.Weights.open(io, allocator, w_path);
    defer weights.close(io);
    weights.prefetch(req.prefetch);
    var model = try model_mod.Model.load(io, allocator, core.nn.defaultBackend(), c, &weights);
    defer model.deinit(allocator);
    const t_loaded: f64 = @as(f64, @floatFromInt(Io.Timestamp.now(io, .real).nanoseconds)) / 1e9;

    const series = try series_io.loadBlocks(io, allocator, req.series_path);
    defer allocator.free(series.data);
    const b = series.blocks;
    const v = series.rows;

    // Chronos-2 v1 is univariate: every (series, variate) row is its own independent series.
    const values = try series_io.toF32(allocator, series.data);
    defer allocator.free(values);
    const out = try forecaster.forecast(allocator, req.par_io, &model, values, b * v, series.len, req.horizon);
    errdefer allocator.free(out);
    const levels = try allocator.dupe(f64, c.chronos_config.quantiles);
    return .{ .levels = levels, .values = out, .series = b, .variates = v, .horizon = req.horizon, .t_loaded = t_loaded };
}
