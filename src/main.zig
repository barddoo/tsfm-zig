//! timestral: zero-shot probabilistic time series forecasting from a checkpoint of any registered
//! model family (see `families.zig`); the family is selected from `config.json`.

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const core = @import("core");
const hub = core.hub;
const par = core.par;
const family = core.family;
const families = @import("families.zig");

const usage =
    \\usage: timestral <model> <series.csv> --horizon N [options]
    \\
    \\  <model>       a directory with config.json and model.safetensors, or a Hugging Face
    \\                Hub repo id such as mistralai/Timestral-28M, amazon/chronos-2 or
    \\                amazon/chronos-bolt-small (cached, else downloaded). The model family is
    \\                selected by its config.json.
    \\  <series.csv>  one line per variate; values separated by commas or whitespace; nan / NA /
    \\                an empty field marks a missing value. Separate several series with a
    \\                blank line (all with the same number of variates and steps).
    \\
    \\Timestral covariates (same blank-line-separated blocks, one block per series, one line
    \\per covariate; Timestral only):
    \\  --past FILE         past-only covariates, as many steps as the series
    \\  --known FILE        known-future covariates, steps + horizon values per line
    \\  --cat-past FILE     categorical codes, as many steps as the series
    \\  --cat-known FILE    categorical codes, steps + horizon values per line
    \\
    \\options (Chronos-2 horizons are limited to 1024 steps; Chronos-2 and Chronos-Bolt series are
    \\univariate and take no covariates or Timestral-only flags):
    \\  --horizon N         steps to forecast (required)
    \\  --max-context N     keep only the most recent N steps (default: trained length)
    \\  --block-patches N   patches decoded per block (default 32)
    \\  --no-symmetric      skip averaging with the reflected forecast of -context
    \\  --no-clip           do not clamp non-negative series at 0
    \\  --prefetch          ask the kernel to read the checkpoint in at load (madvise WILLNEED);
    \\                      on by default with --threads 1, where it speeds up the first forecast
    \\  --no-prefetch       never prefetch (the default with more than one thread)
    \\  --revision REV      Hub branch, tag or commit (default main)
    \\  --offline           never touch the network (also HF_HUB_OFFLINE=1)
    \\  --threads N         worker threads (default: one per logical CPU, or per performance core on Apple Silicon)
    \\  --time              print model load and forecast wall time to stderr
    \\
    \\prints CSV: series,variate,step,q<level>... (step 1 is the first forecast step)
    \\
;

const Args = struct {
    model: []const u8,
    series: []const u8,
    horizon: usize = 0,
    revision: []const u8 = "main",
    offline: bool = false,
    time: bool = false,
    threads: usize = 0, // 0 = one per logical CPU
    covariates: family.CovariatePaths = .{},
    opts: family.Options = .{},
    prefetch: ?core.safetensors.Prefetch = null, // null: decided by the thread count
};

fn parseArgs(argv: []const [:0]const u8, env: *const std.process.Environ.Map) !Args {
    var positional: [2][]const u8 = undefined;
    var npos: usize = 0;
    var args: Args = .{ .model = "", .series = "" };
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a: []const u8 = argv[i];
        const takes_number = mem.eql(u8, a, "--horizon") or mem.eql(u8, a, "--threads") or mem.eql(u8, a, "--max-context") or mem.eql(u8, a, "--block-patches");
        const takes_string = mem.eql(u8, a, "--past") or mem.eql(u8, a, "--known") or mem.eql(u8, a, "--cat-past") or
            mem.eql(u8, a, "--cat-known") or mem.eql(u8, a, "--revision");
        if (takes_number or takes_string) {
            i += 1;
            if (i >= argv.len) return error.MissingValue;
            const sval: []const u8 = argv[i];
            if (takes_number) {
                const val = std.fmt.parseInt(usize, sval, 10) catch return error.InvalidNumber;
                if (mem.eql(u8, a, "--horizon")) args.horizon = val;
                if (mem.eql(u8, a, "--threads")) args.threads = val;
                if (mem.eql(u8, a, "--max-context")) args.opts.max_context = val;
                if (mem.eql(u8, a, "--block-patches")) args.opts.block_patches = val;
            } else if (mem.eql(u8, a, "--past")) {
                args.covariates.past = sval;
            } else if (mem.eql(u8, a, "--known")) {
                args.covariates.known = sval;
            } else if (mem.eql(u8, a, "--cat-past")) {
                args.covariates.cat_past = sval;
            } else if (mem.eql(u8, a, "--cat-known")) {
                args.covariates.cat_known = sval;
            } else args.revision = sval;
        } else if (mem.eql(u8, a, "--no-symmetric")) {
            args.opts.no_symmetric = true;
        } else if (mem.eql(u8, a, "--no-clip")) {
            args.opts.no_clip = true;
        } else if (mem.eql(u8, a, "--prefetch")) {
            args.prefetch = .will_need;
        } else if (mem.eql(u8, a, "--no-prefetch")) {
            args.prefetch = .off;
        } else if (mem.eql(u8, a, "--offline")) {
            args.offline = true;
        } else if (mem.eql(u8, a, "--time")) {
            args.time = true;
        } else if (a.len > 0 and a[0] == '-') {
            return error.UnknownOption;
        } else {
            if (npos == 2) return error.TooManyArguments;
            positional[npos] = argv[i];
            npos += 1;
        }
    }
    if (npos != 2 or args.horizon == 0) return error.MissingArguments;
    args.model = positional[0];
    args.series = positional[1];
    if (env.get("HF_HUB_OFFLINE")) |v| {
        if (mem.eql(u8, v, "1")) args.offline = true;
    }
    return args;
}

fn run(init: std.process.Init) !void {
    const allocator = init.gpa;

    var arg_iter = try init.minimal.args.iterateAllocator(allocator);
    defer arg_iter.deinit();
    var argv: std.ArrayList([:0]const u8) = .empty;
    defer argv.deinit(allocator);
    while (arg_iter.next()) |a| try argv.append(allocator, a);
    const args = parseArgs(argv.items, init.environ_map) catch |err| {
        std.debug.print("argument error: {s}\n", .{@errorName(err)});
        return err;
    };

    // Worker pool for `par.forRange` (this thread plus `threads - 1` workers); also the `Io`
    // behind file access and downloads.
    par.threads = if (args.threads == 0) core.sys.defaultThreads() else args.threads;
    var threaded: Io.Threaded = .init(allocator, .{ .async_limit = .limited(par.threads - 1) });
    defer threaded.deinit();
    const io = threaded.io();
    const par_io: Io = if (par.threads > 1) io else Io.Threaded.global_single_threaded.io();
    const t_start: f64 = @as(f64, @floatFromInt(Io.Timestamp.now(io, .real).nanoseconds)) / 1e9;

    const model_dir = hub.resolve(io, allocator, init.environ_map, args.model, args.revision, args.offline) catch |err| {
        std.debug.print("cannot find model '{s}' ({s})\n", .{ args.model, @errorName(err) });
        return err;
    };
    defer allocator.free(model_dir);

    const cfg_path = try allocator.print("{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    const cfg_text = try Io.Dir.cwd().readFileAlloc(io, cfg_path, allocator, .unlimited);
    defer allocator.free(cfg_text);

    const req: family.Request = .{
        .io = io,
        .par_io = par_io,
        .allocator = allocator,
        .model_dir = model_dir,
        .config_text = cfg_text,
        .series_path = args.series,
        .covariates = args.covariates,
        .horizon = args.horizon,
        .options = args.opts,
        .prefetch = args.prefetch orelse if (par.threads == 1) .will_need else .off,
    };
    const result = families.Registry.run(&req) catch |err| {
        if (err == error.UnsupportedArchitecture) {
            std.debug.print("unsupported model: config.json in '{s}' is not a known family (supported: {s})\n", .{ model_dir, families.Registry.names });
        }
        return err;
    };
    defer result.deinit(allocator);
    if (args.time) {
        const t_done: f64 = @as(f64, @floatFromInt(Io.Timestamp.now(io, .real).nanoseconds)) / 1e9;
        std.debug.print("load={d:.3}s forecast={d:.3}s\n", .{ result.t_loaded - t_start, t_done - result.t_loaded });
    }

    try writeCsv(io, allocator, result);
}

fn writeCsv(io: Io, allocator: mem.Allocator, r: family.Forecast) !void {
    const q = r.levels.len;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "series,variate,step");
    for (r.levels) |level| try out.print(allocator, ",q{d}", .{level});
    try out.append(allocator, '\n');
    for (0..r.series) |s| {
        for (0..r.variates) |v| {
            for (0..r.horizon) |step| {
                try out.print(allocator, "{d},{d},{d}", .{ s, v, step + 1 });
                for (r.values[(((s * r.variates) + v) * r.horizon + step) * q ..][0..q]) |x| try out.print(allocator, ",{d}", .{x});
                try out.append(allocator, '\n');
            }
        }
    }
    try Io.File.stdout().writeStreamingAll(io, out.items);
}

pub fn main(init: std.process.Init) u8 {
    run(init) catch |err| {
        switch (err) {
            error.MissingArguments, error.MissingValue, error.InvalidNumber, error.UnknownOption, error.TooManyArguments => Io.File.stderr().writeStreamingAll(init.io, usage) catch {},
            error.UnsupportedArchitecture => {},
            else => std.debug.print("error: {s}\n", .{@errorName(err)}),
        }
        return 1;
    };
    return 0;
}
