//! Audio processing: decode, resample, encode, mix.
//!
//! All audio stays as f32 slices in memory. No intermediate files,
//! no WAV/FLAC containers. Input is raw PCM, internal representation
//! is mono f32 samples.

pub mod decode;
pub mod encode;
pub mod mix;
pub mod resample;
