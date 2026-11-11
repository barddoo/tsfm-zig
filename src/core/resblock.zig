//! The residual MLP block shared by the Chronos family's patch embeddings and output heads:
//! `output(relu(hidden(x))) + residual(x)`, with biases, as `ResidualBlock` in the reference.

const std = @import("std");
const nn = @import("nn.zig");
const st = @import("safetensors.zig");

const Allocator = std.mem.Allocator;

pub const Linear = struct {
    w: []const f32,
    b: ?[]const f32 = null,
};

fn loadLinear(io: std.Io, w: *st.Weights, buf: []u8, prefix: []const u8, name: []const u8, out: usize, in: usize) !Linear {
    const wn = std.fmt.bufPrint(buf, "{s}.{s}.weight", .{ prefix, name }) catch unreachable;
    const weight = try w.get(io, wn, out * in);
    const bn = std.fmt.bufPrint(buf, "{s}.{s}.bias", .{ prefix, name }) catch unreachable;
    const bias = try w.get(io, bn, out);
    return .{ .w = weight, .b = bias };
}

pub const ResBlock = struct {
    hidden: Linear,
    output: Linear,
    residual: Linear,
    in: usize,
    hid: usize,
    out: usize,

    pub fn load(io: std.Io, w: *st.Weights, buf: []u8, prefix: []const u8, in: usize, hid: usize, out: usize) !ResBlock {
        return .{
            .hidden = try loadLinear(io, w, buf, prefix, "hidden_layer", hid, in),
            .output = try loadLinear(io, w, buf, prefix, "output_layer", out, hid),
            .residual = try loadLinear(io, w, buf, prefix, "residual_layer", out, in),
            .in = in,
            .hid = hid,
            .out = out,
        };
    }

    /// `output(relu(hidden(x))) + residual(x)` for `rows` input vectors.
    pub fn forward(self: ResBlock, io: std.Io, allocator: Allocator, y: []f32, x: []const f32, rows: usize) !void {
        const h = try allocator.alloc(f32, rows * self.hid);
        defer allocator.free(h);
        nn.linear(io, h, x, rows, self.hidden.w, self.hid, self.in, self.hidden.b);
        nn.relu(h);
        nn.linear(io, y, h, rows, self.output.w, self.out, self.hid, self.output.b);
        const r = try allocator.alloc(f32, rows * self.out);
        defer allocator.free(r);
        nn.linear(io, r, x, rows, self.residual.w, self.out, self.in, self.residual.b);
        nn.addInPlace(y[0 .. rows * self.out], r);
    }
};
