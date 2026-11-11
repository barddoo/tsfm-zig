//! Series input: blank-line-separated blocks of equal-length rows, as read from a CSV file.

const std = @import("std");
const sys = @import("sys.zig");

/// A file of blank-line-separated blocks of equal-length rows, as `(blocks, rows, len)` float64.
pub const Blocks = struct {
    data: []f64,
    blocks: usize,
    rows: usize,
    len: usize,
};

fn parseValue(tok: []const u8) f64 {
    const s = std.mem.trim(u8, tok, " \t\r");
    if (s.len == 0) return std.math.nan(f64);
    return std.fmt.parseFloat(f64, s) catch std.math.nan(f64);
}

pub fn parseBlocks(allocator: std.mem.Allocator, text: []const u8) !Blocks {
    var vals: std.ArrayList(f64) = .empty;
    errdefer vals.deinit(allocator);
    var blocks: usize = 0;
    var rows_in_block: usize = 0;
    var rows: usize = 0;
    var len: usize = 0;
    var have_len = false;

    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len > 0 and line[0] == '#') continue;
        if (line.len == 0) {
            // A blank line closes the current block.
            if (rows_in_block > 0) {
                if (blocks == 0) rows = rows_in_block else if (rows_in_block != rows) return error.RaggedBlocks;
                blocks += 1;
                rows_in_block = 0;
            }
            continue;
        }
        const before = vals.items.len;
        if (std.mem.indexOfScalar(u8, line, ',') != null) {
            var f = std.mem.splitScalar(u8, line, ',');
            while (f.next()) |tok| try vals.append(allocator, parseValue(tok));
        } else {
            var f = std.mem.tokenizeAny(u8, line, " \t");
            while (f.next()) |tok| try vals.append(allocator, parseValue(tok));
        }
        const n = vals.items.len - before;
        if (!have_len) {
            len = n;
            have_len = true;
        } else if (n != len) {
            return error.RaggedInput;
        }
        rows_in_block += 1;
    }
    if (rows_in_block > 0) {
        if (blocks == 0) rows = rows_in_block else if (rows_in_block != rows) return error.RaggedBlocks;
        blocks += 1;
    }
    if (blocks == 0 or len == 0) return error.EmptyInput;
    return .{ .data = try vals.toOwnedSlice(allocator), .blocks = blocks, .rows = rows, .len = len };
}

pub fn loadBlocks(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Blocks {
    const text = try sys.readFile(io, allocator, path);
    defer allocator.free(text);
    return parseBlocks(allocator, text);
}

pub fn toF32(allocator: std.mem.Allocator, x: []const f64) ![]f32 {
    const out = try allocator.alloc(f32, x.len);
    for (out, x) |*o, v| o.* = @floatCast(v);
    return out;
}

/// A covariate file, checked against the series: one block per series, rows of `len` values.
pub fn covariateFile(io: std.Io, allocator: std.mem.Allocator, path: ?[]const u8, batch: usize, len: usize, what: []const u8) !?Blocks {
    const p = path orelse return null;
    const blk = try loadBlocks(io, allocator, p);
    if (blk.blocks != batch or blk.len != len) {
        sys.eprintf(io, "{s}: expected {d} series with {d} values per line, got {d} series with {d}\n", .{ what, batch, len, blk.blocks, blk.len });
        return error.CovariateShape;
    }
    return blk;
}
