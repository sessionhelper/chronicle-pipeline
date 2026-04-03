//! Audio encoding (future).
//!
//! Will handle f32 -> Opus/OGG encoding for serving audio to browsers.
//! Gated behind the `opus` feature flag.

// TODO: Implement Opus encoding when the `opus` feature is added.
// - Accept f32 mono samples at a given sample rate
// - Encode to Opus frames
// - Wrap in OGG container
// - Return encoded bytes
