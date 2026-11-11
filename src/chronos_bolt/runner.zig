//! Chronos-Bolt as a registered model family (univariate, no covariates).

const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const model_mod = @import("model.zig");
const forecaster = @import("forecaster.zig");

const family = core.family;
const series_io = core.series;
const sys = core.sys;
const st = core.safetensors;

pub const name = "Chronos-Bolt";

pub fn matches(allocator: std.mem.Allocator, config_text: []const u8) !bool {
    return config.isChronosBolt(allocator, config_text);
}

pub fn run(req: *const family.Request) !family.Forecast {
    const allocator = req.allocator;
    const io = req.io;

    if (req.covariates.any()) {
        sys.eprint(io, "Chronos-Bolt does not support covariates: --past, --known, --cat-past and --cat-known are unavailable\n");
        return error.CovariatesUnsupported;
    }
    const o = req.options;
    if (o.max_context != null or o.block_patches != null or o.no_symmetric or o.no_clip) {
        sys.eprint(io, "--max-context, --block-patches, --no-symmetric and --no-clip are Timestral-only options\n");
        return error.UnsupportedOption;
    }

    const parsed = try config.parse(allocator, req.config_text);
    defer parsed.deinit();
    const c = parsed.value;

    const w_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{req.model_dir});
    defer allocator.free(w_path);
    var weights = try st.Weights.open(io, allocator, w_path);
    defer weights.close(io);
    weights.prefetch(req.prefetch);
    var model = try model_mod.Model.load(io, allocator, c, &weights);
    defer model.deinit(allocator);
    const t_loaded = sys.now(io);

    const series = try series_io.loadBlocks(io, allocator, req.series_path);
    defer allocator.free(series.data);
    const b = series.blocks;
    const v = series.rows;

    // Every (series, variate) row is its own independent series.
    const values = try series_io.toF32(allocator, series.data);
    defer allocator.free(values);
    const out = try forecaster.forecast(allocator, req.par_io, &model, values, b * v, series.len, req.horizon);
    errdefer allocator.free(out);
    const levels = try allocator.dupe(f64, c.chronos_config.quantiles);
    return .{ .levels = levels, .values = out, .series = b, .variates = v, .horizon = req.horizon, .t_loaded = t_loaded };
}
