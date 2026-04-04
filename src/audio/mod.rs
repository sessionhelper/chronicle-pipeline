//! Audio processing: resample, encode, mix.
//!
//! All audio stays as f32 slices in memory. Input to the pipeline
//! is already mono f32 samples — byte decoding and stereo downmix
//! are the caller's responsibility.

pub mod encode;
pub mod mix;
pub mod resample;
