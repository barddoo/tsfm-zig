//! Memory mapping helpers over `Io`.

const std = @import("std");
const Io = std.Io;
const builtin = @import("builtin");

/// How the kernel should expect a mapping to be read (`madvise`). Only a hint.
pub const Advice = enum {
    /// Start reading the whole range in now (`MADV_WILLNEED`). On macOS the call returns once the
    /// pages are resident, so it blocks for roughly the time of a sequential read of the file.
    will_need,
};

/// A read-only memory mapping of a whole file.
pub const Mapped = struct {
    file: Io.File,
    map: Io.File.MemoryMap,
    bytes: []const u8,

    pub fn open(io: Io, path: []const u8) !Mapped {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        const map = try Io.File.MemoryMap.create(io, file, .{
            .len = len,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        });
        std.debug.assert(len <= map.memory.len);
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

    pub fn close(self: *Mapped, io: Io) void {
        self.map.destroy(io);
        self.file.close(io);
    }
};
