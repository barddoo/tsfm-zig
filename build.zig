const std = @import("std");
const Translator = @import("translate_c").Translator;

const Blas = enum { accelerate, openblas, builtin };
const Gpu = enum { none, metal, cuda };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_blas: Blas = if (target.result.os.tag == .macos) .accelerate else .builtin;
    const blas = b.option(Blas, "blas", "Matrix-product backend: accelerate (macOS), openblas, or builtin SIMD kernels") orelse default_blas;

    // GPU is opt-in: it duplicates the GEMM weights in private memory, which the CPU build avoids.
    const gpu = b.option(Gpu, "gpu", "GPU backend: metal (macOS), cuda (Linux/NVIDIA) or none") orelse .none;
    if (gpu == .metal and target.result.os.tag != .macos) {
        @panic("-Dgpu=metal requires a macOS target");
    }
    if (gpu == .cuda and target.result.os.tag != .linux) {
        @panic("-Dgpu=cuda currently requires a Linux target");
    }

    // The `c` module: the BLAS header translated to Zig, or an empty module for the builtin
    // kernels (which then need no C library at all).
    const c_mod = switch (blas) {
        .builtin => b.createModule(.{ .root_source_file = b.path("src/core/c_none.zig") }),
        .accelerate, .openblas => blk: {
            const header = b.addWriteFiles().add("c.h", switch (blas) {
                .accelerate => "#include <Accelerate/Accelerate.h>\n",
                else => "#include <cblas.h>\n",
            });
            const translate_c = b.dependency("translate_c", .{});
            const c: Translator = .init(translate_c, .{
                .c_source_file = header,
                .target = target,
                .optimize = optimize,
            });
            break :blk c.mod;
        },
    };

    // The `backend` module is the runtime vtable shared by `core` and the selected GPU module
    // (a source file may belong to only one Zig module).
    const backend_mod = b.createModule(.{ .root_source_file = b.path("src/platform/backend.zig") });

    // The `gpu` module: backend declarations, or a no-op stub.
    const gpu_mod = switch (gpu) {
        .none => b.createModule(.{
            .root_source_file = b.path("src/platform/none/gpu.zig"),
            .imports = &.{.{ .name = "backend", .module = backend_mod }},
        }),
        .metal => b.createModule(.{
            .root_source_file = b.path("src/platform/metal/gpu.zig"),
            .imports = &.{.{ .name = "backend", .module = backend_mod }},
        }),
        .cuda => b.createModule(.{
            .root_source_file = b.path("src/platform/cuda/gpu.zig"),
            .imports = &.{.{ .name = "backend", .module = backend_mod }},
        }),
    };

    // Shared kernels and IO, imported by both model implementations.
    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/core/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = blas != .builtin or gpu != .none,
        .imports = &.{
            .{ .name = "c", .module = c_mod },
            .{ .name = "gpu", .module = gpu_mod },
            .{ .name = "backend", .module = backend_mod },
        },
    });
    switch (blas) {
        .accelerate => core_mod.linkFramework("Accelerate", .{}),
        .openblas => core_mod.linkSystemLibrary("openblas", .{}),
        .builtin => {},
    }
    switch (gpu) {
        .metal => {
            core_mod.linkFramework("Metal", .{});
            core_mod.linkFramework("MetalPerformanceShaders", .{});
            core_mod.addCSourceFile(.{
                .file = b.path("src/platform/metal/gemm.m"),
                .flags = &.{"-fobjc-arc"},
            });
            core_mod.linkSystemLibrary("objc", .{});
        },
        .cuda => {
            core_mod.addCSourceFile(.{
                .file = b.path("src/platform/cuda/gemm.c"),
                .flags = &.{"-O2"},
            });
            core_mod.linkSystemLibrary("dl", .{});
        },
        .none => {},
    }

    const needs_libc = blas != .builtin or gpu != .none;

    const chronos2_mod = b.addModule("chronos2", .{
        .root_source_file = b.path("src/chronos2/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = needs_libc,
        .imports = &.{.{ .name = "core", .module = core_mod }},
    });

    const bolt_mod = b.addModule("chronos_bolt", .{
        .root_source_file = b.path("src/chronos_bolt/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = needs_libc,
        .imports = &.{.{ .name = "core", .module = core_mod }},
    });

    const exe = b.addExecutable(.{
        .name = "timestral",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = needs_libc,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "chronos2", .module = chronos2_mod },
                .{ .name = "chronos_bolt", .module = bolt_mod },
            },
        }),
    });
    switch (blas) {
        .accelerate => exe.root_module.linkFramework("Accelerate", .{}),
        .openblas => exe.root_module.linkSystemLibrary("openblas", .{}),
        .builtin => {},
    }
    switch (gpu) {
        .metal => {
            exe.root_module.linkFramework("Metal", .{});
            exe.root_module.linkFramework("MetalPerformanceShaders", .{});
        },
        .none, .cuda => {},
    }
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the forecaster").dependOn(&run_cmd.step);

    // Library module for downstream consumers (e.g. the HTTP server in the forecast service).
    const lib_mod = b.addModule("timestral", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = needs_libc,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "chronos2", .module = chronos2_mod },
            .{ .name = "chronos_bolt", .module = bolt_mod },
        },
    });
    switch (blas) {
        .accelerate => lib_mod.linkFramework("Accelerate", .{}),
        .openblas => lib_mod.linkSystemLibrary("openblas", .{}),
        .builtin => {},
    }
    switch (gpu) {
        .metal => {
            lib_mod.linkFramework("Metal", .{});
            lib_mod.linkFramework("MetalPerformanceShaders", .{});
        },
        .none, .cuda => {},
    }

    // Unit tests: the attention kernel and its masks, the registry, series parsing and the pieces of
    // Chronos-Bolt that have a closed-form reference. Model parity against the PyTorch references
    // needs checkpoints and is run by hand (see tests/).
    const test_step = b.step("test", "Run unit tests");
    for ([_]*std.Build.Module{ core_mod, bolt_mod }) |m| {
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
