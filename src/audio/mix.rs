//! Track mixing (future).
//!
//! Will combine multiple speaker tracks into a single mixed stream.
//! Useful for generating a combined playback track.

// TODO: Implement track mixing.
// - Accept multiple SpeakerSamples at the same sample rate
// - Align by timestamp and sum/average overlapping samples
// - Apply normalization to prevent clipping
// - Return a single mixed SpeakerSamples
