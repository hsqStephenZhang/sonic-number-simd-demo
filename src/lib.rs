#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon;
#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
pub mod sve;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "avx2",
    target_feature = "sse2"
))]
pub mod x86_64;
