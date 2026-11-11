//! Model-family interface and a comptime registry.
//!
//! A family is a struct (usually a file) with these declarations:
//!
//!     pub const name: []const u8;
//!     pub fn matches(allocator: Allocator, config_text: []const u8) !bool;
//!     pub fn run(req: *const Request) !Forecast;
//!
//! `Registry(.{ a, b, c })` checks every family at compile time and dispatches with an unrolled
//! `inline for`, so adding a family is one entry in that tuple. Nothing is selected by default:
//! a `config.json` that no family claims is an error, not a guess.

const std = @import("std");
const safetensors = @import("safetensors.zig");

/// Covariate files (Timestral-style); a family that has no covariates rejects any that are set.
pub const CovariatePaths = struct {
    past: ?[]const u8 = null,
    known: ?[]const u8 = null,
    cat_past: ?[]const u8 = null,
    cat_known: ?[]const u8 = null,

    pub fn any(self: CovariatePaths) bool {
        return self.past != null or self.known != null or self.cat_past != null or self.cat_known != null;
    }
};

/// Tuning flags. `null` / `false` means "not given", so a family can reject exactly the flags it
/// does not implement.
pub const Options = struct {
    max_context: ?usize = null,
    block_patches: ?usize = null,
    no_symmetric: bool = false,
    no_clip: bool = false,
};

pub const Request = struct {
    io: std.Io,
    /// `io` when multi-threaded, else the single-threaded `Io`; handed to the layers.
    par_io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config_text: []const u8,
    series_path: []const u8,
    covariates: CovariatePaths = .{},
    horizon: usize,
    options: Options = .{},
    /// Read-ahead hint applied to the checkpoint right after it is mapped.
    prefetch: safetensors.Prefetch = .off,
};

/// `(series, variates, horizon, levels.len)` quantile forecasts. Owns `levels` and `values`.
pub const Forecast = struct {
    levels: []f64,
    values: []f32,
    series: usize,
    variates: usize,
    horizon: usize,
    /// `sys.now` once the weights were loaded and just before forecasting starts.
    t_loaded: f64,

    pub fn deinit(self: Forecast, allocator: std.mem.Allocator) void {
        allocator.free(self.levels);
        allocator.free(self.values);
    }
};

fn checkFamily(comptime F: type) void {
    if (!@hasDecl(F, "name")) @compileError("model family is missing `pub const name`");
    if (!@hasDecl(F, "matches")) @compileError("model family '" ++ F.name ++ "' is missing `matches`");
    if (!@hasDecl(F, "run")) @compileError("model family '" ++ F.name ++ "' is missing `run`");
    const m = @typeInfo(@TypeOf(F.matches)).@"fn";
    if (m.param_types.len != 2 or m.param_types[0] != std.mem.Allocator or m.param_types[1] != []const u8)
        @compileError("'" ++ F.name ++ "'.matches must be fn (Allocator, []const u8) !bool");
    const r = @typeInfo(@TypeOf(F.run)).@"fn";
    if (r.param_types.len != 1 or r.param_types[0] != *const Request)
        @compileError("'" ++ F.name ++ "'.run must be fn (*const Request) !Forecast");
}

pub fn Registry(comptime families: anytype) type {
    comptime for (families) |F| checkFamily(F);

    return struct {
        /// Comma-separated family names, for messages.
        pub const names: []const u8 = blk: {
            var s: []const u8 = "";
            for (families, 0..) |F, i| s = s ++ (if (i == 0) "" else ", ") ++ F.name;
            break :blk s;
        };

        /// The name of the family that claims this `config.json`, or null.
        pub fn detect(allocator: std.mem.Allocator, config_text: []const u8) !?[]const u8 {
            inline for (families) |F| {
                if (try F.matches(allocator, config_text)) return F.name;
            }
            return null;
        }

        /// Run the family that claims `req.config_text`; `error.UnsupportedArchitecture` if none does.
        pub fn run(req: *const Request) !Forecast {
            inline for (families) |F| {
                if (try F.matches(req.allocator, req.config_text)) return F.run(req);
            }
            return error.UnsupportedArchitecture;
        }
    };
}

/// Whether `config.json` lists `architecture` in its `architectures` array. A convenience for
/// families that are identified that way.
pub fn hasArchitecture(allocator: std.mem.Allocator, config_text: []const u8, architecture: []const u8) !bool {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config_text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const arch = parsed.value.object.get("architectures") orelse return false;
    if (arch != .array) return false;
    for (arch.array.items) |item| {
        if (item == .string and std.mem.eql(u8, item.string, architecture)) return true;
    }
    return false;
}
