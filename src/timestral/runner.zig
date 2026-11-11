//! Timestral as a registered model family: loads the checkpoint, reads the series and covariate
//! files and runs the block-autoregressive forecaster.

const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const model_mod = @import("model.zig");
const forecaster = @import("forecaster.zig");

const family = core.family;
const series_io = core.series;
const sys = core.sys;
const st = core.safetensors;

pub const name = "Timestral";

pub fn matches(allocator: std.mem.Allocator, config_text: []const u8) !bool {
    return cfg.isTimestral(allocator, config_text);
}

pub fn run(req: *const family.Request) !family.Forecast {
    const allocator = req.allocator;
    const io = req.io;

    const parsed = try cfg.parse(allocator, req.config_text);
    defer parsed.deinit();

    const w_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{req.model_dir});
    defer allocator.free(w_path);
    var weights = try st.Weights.open(io, allocator, w_path);
    defer weights.close(io);
    weights.prefetch(req.prefetch);
    var model = try model_mod.Model.load(io, allocator, core.nn.defaultBackend(), parsed.value, &weights);
    defer model.deinit(allocator);
    const t_loaded = sys.now(io);

    const series = try series_io.loadBlocks(io, allocator, req.series_path);
    defer allocator.free(series.data);
    const b = series.blocks;
    const v = series.rows;
    const t = series.len;

    const values = try series_io.toF32(allocator, series.data);
    defer allocator.free(values);
    const observed = try allocator.alloc(bool, values.len);
    defer allocator.free(observed);
    for (values, observed) |*x, *o| {
        o.* = std.math.isFinite(x.*);
        if (!o.*) x.* = 0;
    }

    const cp = req.covariates;
    var cov: forecaster.Covariates = .{};
    const past = try series_io.covariateFile(io, allocator, cp.past, b, t, "--past");
    const known = try series_io.covariateFile(io, allocator, cp.known, b, t + req.horizon, "--known");
    const cat_past = try series_io.covariateFile(io, allocator, cp.cat_past, b, t, "--cat-past");
    const cat_known = try series_io.covariateFile(io, allocator, cp.cat_known, b, t + req.horizon, "--cat-known");
    defer for ([_]?series_io.Blocks{ past, known, cat_past, cat_known }) |blk| if (blk) |x| allocator.free(x.data);
    const past32 = if (past) |p| try series_io.toF32(allocator, p.data) else null;
    defer if (past32) |x| allocator.free(x);
    const known32 = if (known) |k| try series_io.toF32(allocator, k.data) else null;
    defer if (known32) |x| allocator.free(x);
    if (past) |p| {
        cov.past = past32;
        cov.n_past = p.rows;
    }
    if (known) |k| {
        cov.known = known32;
        cov.n_known = k.rows;
    }
    if (cat_past) |p| {
        cov.cat_past = p.data;
        cov.n_cat_past = p.rows;
    }
    if (cat_known) |k| {
        cov.cat_known = k.data;
        cov.n_cat_known = k.rows;
    }

    var opts: forecaster.Options = .{};
    if (req.options.max_context) |n| opts.max_context = n;
    if (req.options.block_patches) |n| opts.per_shot_horizon_patches = @max(n, 1);
    if (req.options.no_symmetric) opts.symmetric_averaging = false;
    if (req.options.no_clip) opts.clip_nonnegative = false;

    const values_out = try forecaster.forecast(allocator, req.par_io, &model, values, observed, b, v, t, req.horizon, cov, opts);
    errdefer allocator.free(values_out);
    const levels = try allocator.dupe(f64, parsed.value.quantile_knots);
    return .{ .levels = levels, .values = values_out, .series = b, .variates = v, .horizon = req.horizon, .t_loaded = t_loaded };
}
