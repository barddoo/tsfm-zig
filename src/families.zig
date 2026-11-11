//! The registered model families. Adding a family is one entry in this tuple (plus its module);
//! every entry is checked against the `core.family` interface at compile time.

const core = @import("core");
const chronos2 = @import("chronos2");
const chronos_bolt = @import("chronos_bolt");
const timestral = @import("timestral/runner.zig");

pub const Registry = core.family.Registry(.{ timestral, chronos2.runner, chronos_bolt.runner });
