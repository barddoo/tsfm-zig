//! Memory-mapped safetensors reader. F32 tensors are used in place; F16 and BF16 tensors are
//! converted to float32 on first access.

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const sys = @import("sys.zig");

/// Whether and how to ask the kernel to read the checkpoint ahead of the forward pass.
///
/// The weights are an mmap, so by default a cold file is read by page faults as the first forecast
/// touches each tensor. Hinting the whole mapping lets the kernel read it in large sequential
/// requests instead. It changes only when pages are read, never the values.
///
/// Measured on macOS/APFS with a cold file: with one thread the first forecast is about 1.9x faster
/// end to end (the read moves into this call); with a worker pool the unhinted parallel faults are
/// already efficient, so the gain is small and can turn into a small loss. It is therefore opt-in.
/// `MADV_SEQUENTIAL` was measured too and was slower in every case, so it is not offered.
pub const Prefetch = enum {
    off,
    /// `MADV_WILLNEED` on the whole mapping.
    will_need,
};

pub const Weights = struct {
    allocator: mem.Allocator,
    converted: std.ArrayList([]f32) = .empty,
    map: sys.Mapped,
    header: std.json.Parsed(std.json.Value),
    data_start: usize,

    pub fn open(io: Io, allocator: mem.Allocator, path: []const u8) !Weights {
        var map = try sys.Mapped.open(io, path);
        errdefer map.close(io);
        if (map.bytes.len < 8) return error.InvalidSafetensors;
        const hlen: usize = @intCast(mem.readInt(u64, map.bytes[0..8], .little));
        if (8 + hlen > map.bytes.len) return error.InvalidSafetensors;
        std.debug.assert(8 + hlen <= map.bytes.len);
        const header = try std.json.parseFromSlice(std.json.Value, allocator, map.bytes[8 .. 8 + hlen], .{});
        return .{ .allocator = allocator, .map = map, .header = header, .data_start = 8 + hlen };
    }

    /// Apply `mode` to the mapped checkpoint. Call once after `open`; safe to skip.
    pub fn prefetch(self: *const Weights, mode: Prefetch) void {
        switch (mode) {
            .off => {},
            .will_need => self.map.advise(.will_need),
        }
    }

    pub fn close(self: *Weights, io: Io) void {
        for (self.converted.items) |buf| self.allocator.free(buf);
        self.converted.deinit(self.allocator);
        self.header.deinit();
        self.map.close(io);
    }

    /// The named float32 tensor, checked against the expected element count.
    pub fn get(self: *Weights, io: Io, name: []const u8, numel: usize) ![]const f32 {
        var err_buf: [256]u8 = undefined;
        const entry = self.header.value.object.get(name) orelse {
            Io.File.stderr().writeStreamingAll(io, mem.print(&err_buf, "missing tensor: {s}\n", .{name}) catch "") catch {};
            return error.MissingTensor;
        };
        const obj = entry.object;
        const dtype = obj.get("dtype").?.string;
        const offs = obj.get("data_offsets").?.array.items;
        const begin: usize = @intCast(offs[0].integer);
        const end: usize = @intCast(offs[1].integer);
        std.debug.assert(begin <= end);
        const elem: usize = if (mem.eql(u8, dtype, "F32")) 4 else if (mem.eql(u8, dtype, "F16") or mem.eql(u8, dtype, "BF16")) 2 else {
            Io.File.stderr().writeStreamingAll(io, mem.print(&err_buf, "tensor {s} has unsupported dtype {s}\n", .{ name, dtype }) catch "") catch {};
            return error.UnsupportedDtype;
        };
        if (end - begin != numel * elem) {
            Io.File.stderr().writeStreamingAll(io, mem.print(&err_buf, "tensor {s}: expected {d} elements, file has {d}\n", .{ name, numel, (end - begin) / elem }) catch "") catch {};
            return error.ShapeMismatch;
        }
        const bytes = self.map.bytes[self.data_start + begin ..][0 .. end - begin];
        std.debug.assert(bytes.len == numel * elem);
        if (elem == 2) {
            const buf = try self.allocator.alloc(f32, numel);
            std.debug.assert(buf.len == numel);
            errdefer self.allocator.free(buf);
            const is_bf16 = dtype[0] == 'B';
            for (buf, 0..) |*o, i| {
                const bits = mem.readInt(u16, bytes[2 * i ..][0..2], .little);
                o.* = if (is_bf16) @bitCast(@as(u32, bits) << 16) else @floatCast(@as(f16, @bitCast(bits)));
            }
            try self.converted.append(self.allocator, buf);
            return buf;
        }
        const addr = @intFromPtr(bytes.ptr);
        if (addr % 4 != 0) return error.MisalignedTensor;
        const p: [*]const f32 = @ptrFromInt(addr);
        return p[0..numel];
    }
};
