//! NEON optimizations for ARM64

#[cfg(target_arch = "aarch64")]
pub fn neon_enabled() -> bool {
    true // NEON is standard on ARM64
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_enabled() -> bool {
    false
}