//! No-op GPU backend (`-Dgpu=none`, the default).

const backend = @import("backend");

pub const has_gpu = false;
pub const name = "none";

pub fn context() ?backend.Backend {
    return null;
}
