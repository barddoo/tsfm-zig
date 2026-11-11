//! The residual MLP block shared by the Chronos and Timestral families' patch embeddings and
//! output heads: `output(act(hidden(x))) + residual(x)`, with optional biases, as `ResidualBlock`
//! in the reference. `act` is a comptime parameter so each variant compiles to its own loop.

const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const nn = @import("nn.zig");
const st = @import("safetensors.zig");

const Allocator = mem.Allocator;

pub const Act = enum { relu, silu };

pub const Linear = struct {
    w: nn.Weight = .{},
    b: ?[]const f32 = null,

    pub fn deinit(self: *const Linear) void {
        self.w.deinit();
    }
};

fn loadLinear(io: Io, backend: ?nn.Backend, w: *st.Weights, buf: []u8, prefix: []const u8, name: []const u8, out: usize, in: usize, bias: bool) !Linear {
    const wn = mem.print(buf, "{s}.{s}.weight", .{ prefix, name }) catch unreachable;
    const weight = try w.get(io, wn, out * in);
    var bv: ?[]const f32 = null;
    if (bias) {
        const bn = mem.print(buf, "{s}.{s}.bias", .{ prefix, name }) catch unreachable;
        bv = try w.get(io, bn, out);
    }
    return .{ .w = nn.Weight.initWith(backend, weight), .b = bv };
}

pub fn ResBlock(comptime act: Act) type {
    return struct {
        hidden: Linear = .{},
        output: Linear = .{},
        residual: Linear = .{},
        in: usize = 0,
        hid: usize = 0,
        out: usize = 0,

        pub fn load(io: Io, backend: ?nn.Backend, w: *st.Weights, buf: []u8, prefix: []const u8, in: usize, hid: usize, out: usize, bias: bool) !@This() {
            var block: @This() = .{ .in = in, .hid = hid, .out = out };
            errdefer block.deinit();
            block.hidden = try loadLinear(io, backend, w, buf, prefix, "hidden_layer", hid, in, bias);
            block.output = try loadLinear(io, backend, w, buf, prefix, "output_layer", out, hid, bias);
            block.residual = try loadLinear(io, backend, w, buf, prefix, "residual_layer", out, in, bias);
            return block;
        }

        pub fn deinit(self: *const @This()) void {
            self.hidden.deinit();
            self.output.deinit();
            self.residual.deinit();
        }

        /// `output(act(hidden(x))) + residual(x)` for `rows` input vectors.
        pub fn forward(self: @This(), io: Io, allocator: Allocator, y: []f32, x: []const f32, rows: usize) !void {
            const h = try allocator.alloc(f32, rows * self.hid);
            defer allocator.free(h);
            nn.linear(io, h, x, rows, self.hidden.w, self.hid, self.in, self.hidden.b);
            switch (act) {
                .relu => nn.relu(h),
                .silu => nn.siluInPlace(h),
            }
            nn.linear(io, y, h, rows, self.output.w, self.out, self.hid, self.output.b);
            const r = try allocator.alloc(f32, rows * self.out);
            defer allocator.free(r);
            nn.linear(io, r, x, rows, self.residual.w, self.out, self.in, self.residual.b);
            nn.addInPlace(y[0 .. rows * self.out], r);
        }
    };
}
