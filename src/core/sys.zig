//! File and process helpers over `std.Io`.

const std = @import("std");
const builtin = @import("builtin");

pub fn eprint(io: std.Io, msg: []const u8) void {
    std.Io.File.stderr().writeStreamingAll(io, msg) catch {};
}

pub fn eprintf(io: std.Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [1024]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    eprint(io, s);
}

pub fn writeStdout(io: std.Io, bytes: []const u8) !void {
    try std.Io.File.stdout().writeStreamingAll(io, bytes);
}

/// The whole file at `path`; the caller frees it.
pub fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

pub fn exists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// How the kernel should expect a mapping to be read (`madvise`). Only a hint.
pub const Advice = enum {
    /// Start reading the whole range in now (`MADV_WILLNEED`). On macOS the call returns once the
    /// pages are resident, so it blocks for roughly the time of a sequential read of the file.
    will_need,
};

/// A read-only memory mapping of a whole file.
pub const Mapped = struct {
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    bytes: []const u8,

    pub fn open(io: std.Io, path: []const u8) !Mapped {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        const map = try std.Io.File.MemoryMap.create(io, file, .{
            .len = len,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        });
        return .{ .file = file, .map = map, .bytes = map.memory[0..len] };
    }

    /// Pass `advice` for the whole mapping to the kernel. Best effort: a hint the system declines or
    /// does not implement is not an error, and the data is read the same either way.
    pub fn advise(self: *const Mapped, advice: Advice) void {
        if (builtin.os.tag == .windows or builtin.os.tag == .wasi) return;
        const flag: u32 = switch (advice) {
            .will_need => std.posix.MADV.WILLNEED,
        };
        std.posix.madvise(self.map.memory.ptr, self.map.memory.len, flag) catch {};
    }

    pub fn close(self: *Mapped, io: std.Io) void {
        self.map.destroy(io);
        self.file.close(io);
    }
};

/// Seconds on a monotonic-enough wall clock, for timing.
pub fn now(io: std.Io) f64 {
    const ns = std.Io.Timestamp.now(io, .real).nanoseconds;
    return @as(f64, @floatFromInt(ns)) / 1e9;
}
