// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared utilities: attention masks, single-position intervention payloads,
//! character-to-token positioning, PCA, seeded Gaussian sampling, and the
//! frozen seeded generator behind it.

#[cfg(any(
    feature = "transformer",
    feature = "rwkv",
    feature = "diffusion",
    feature = "clt",
    feature = "sae"
))]
pub mod inject;
pub mod masks;
pub mod pca;
pub mod positioning;
#[cfg(any(feature = "transformer", feature = "diffusion"))]
pub mod randn;
#[cfg(any(feature = "transformer", feature = "diffusion"))]
pub mod rng;
#[cfg(any(feature = "clt", feature = "sae"))]
pub mod safetensors_view;
